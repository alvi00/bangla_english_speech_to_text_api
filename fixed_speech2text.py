from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from banglaspeech2text import Speech2Text
import torch
import io
import time
import threading
import numpy as np
from collections import deque
import logging
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# Check GPU availability
logger.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize Speech2Text
stt = Speech2Text(
    model_size_or_path="openai/whisper-small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)

class SimpleAudioBuffer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = deque()
        self.lock = threading.Lock()
        self.min_duration = 2.0  # Minimum 2 seconds before processing
        self.max_duration = 8.0  # Maximum 8 seconds
        
    def add_audio(self, audio_data):
        with self.lock:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.buffer.extend(audio_array)
            
            # Keep buffer size manageable
            max_samples = int(self.max_duration * self.sample_rate)
            while len(self.buffer) > max_samples:
                self.buffer.popleft()
    
    def get_audio_for_processing(self):
        with self.lock:
            if len(self.buffer) < int(self.min_duration * self.sample_rate):
                return None
            
            # Get recent audio (last 4 seconds)
            duration = 4.0
            samples_needed = int(duration * self.sample_rate)
            
            if len(self.buffer) >= samples_needed:
                # Get the most recent audio
                audio_data = list(self.buffer)[-samples_needed:]
                return np.array(audio_data, dtype=np.int16).tobytes()
            else:
                # Get all available audio
                audio_data = list(self.buffer)
                return np.array(audio_data, dtype=np.int16).tobytes()
    
    def clear_old_audio(self):
        """Clear processed audio to avoid reprocessing"""
        with self.lock:
            # Keep only the last 2 seconds
            keep_samples = int(2.0 * self.sample_rate)
            if len(self.buffer) > keep_samples:
                # Remove older samples
                to_remove = len(self.buffer) - keep_samples
                for _ in range(to_remove):
                    if self.buffer:
                        self.buffer.popleft()

# Global variables
audio_buffer = SimpleAudioBuffer()
stop_event = threading.Event()
processing_thread = None
last_transcription = ""
last_process_time = 0

def simple_voice_activity_detection(audio_data, threshold=0.01):
    """Simple energy-based voice activity detection"""
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float))))
        zcr = zero_crossings / len(audio_float)
        
        # Voice activity if energy is above threshold and has enough variation
        has_voice = rms > threshold and zcr > 0.01
        
        logger.info(f"Voice activity - RMS: {rms:.4f}, ZCR: {zcr:.4f}, Voice: {has_voice}")
        return has_voice
        
    except Exception as e:
        logger.error(f"VAD error: {e}")
        return True  # Assume voice if detection fails

def process_audio_simple():
    """Simplified audio processing"""
    global last_transcription, last_process_time
    logger.info("Simple audio processing thread started")
    
    while not stop_event.is_set():
        try:
            current_time = time.time()
            
            # Process every 3 seconds
            if current_time - last_process_time < 3.0:
                time.sleep(0.1)
                continue
            
            audio_data = audio_buffer.get_audio_for_processing()
            if audio_data is None:
                time.sleep(0.5)
                continue
            
            # Simple voice activity detection
            if not simple_voice_activity_detection(audio_data):
                logger.info("No voice activity detected, skipping...")
                time.sleep(1.0)
                continue
            
            logger.info(f"Processing {len(audio_data)} bytes of audio")
            
            try:
                # Create audio segment
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                
                # Normalize audio
                audio_segment = audio_segment.normalize()
                
                # Export to WAV
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                # Transcribe
                logger.info("Starting transcription...")
                result = stt.recognize(wav_io, language=None)  # Auto-detect language
                
                if result and hasattr(result, '__iter__'):
                    # Handle if result is iterable (segments)
                    try:
                        segments = list(result)
                        if segments:
                            current_text = " ".join([seg.text.strip() for seg in segments if hasattr(seg, 'text')])
                        else:
                            current_text = ""
                    except:
                        # If it's a simple string result
                        current_text = str(result).strip()
                else:
                    current_text = str(result).strip() if result else ""
                
                logger.info(f"Transcription result: '{current_text}'")
                
                # Only emit if we have meaningful text and it's different
                if current_text and len(current_text.strip()) > 0:
                    # Simple duplicate check
                    if current_text != last_transcription:
                        timestamp = time.strftime('%H:%M:%S')
                        output = f"[{timestamp}] {current_text}"
                        
                        socketio.emit("transcription", {
                            "text": output,
                            "status": "🎙️ Listening...",
                            "is_final": True
                        })
                        
                        logger.info(f"Emitted transcription: {current_text}")
                        last_transcription = current_text
                    else:
                        logger.info("Duplicate transcription, skipping...")
                else:
                    logger.info("Empty transcription result")
                
            except Exception as transcription_error:
                logger.error(f"Transcription error: {transcription_error}")
                socketio.emit("transcription", {
                    "text": f"[ERROR] Transcription failed: {str(transcription_error)}",
                    "status": "❌ Error occurred",
                    "is_final": False
                })
            
            # Clean up old audio
            audio_buffer.clear_old_audio()
            last_process_time = current_time
            
            # Brief pause before next processing
            time.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            time.sleep(2.0)
    
    logger.info("Audio processing thread stopped")

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speech-to-Text (বাংলা + English)</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #transcription { 
            width: 100%; 
            height: 300px; 
            border: 2px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            font-family: monospace;
        }
        #status { 
            margin: 15px 0; 
            padding: 10px;
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }
        button { 
            padding: 10px 20px; 
            margin: 5px; 
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        #start { background: #4CAF50; color: white; }
        #stop { background: #f44336; color: white; }
        #clear { background: #ff9800; color: white; }
        #start:disabled, #stop:disabled { background: #ccc; cursor: not-allowed; }
        .controls { text-align: center; margin: 20px 0; }
        h1 { text-align: center; color: #333; }
        #connection-status { 
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            background: #ffebee;
            color: #c62828;
            margin-bottom: 20px;
        }
        #connection-status.connected { 
            background: #e8f5e8;
            color: #2e7d32;
        }
        .debug { 
            background: #f5f5f5; 
            padding: 10px; 
            margin: 10px 0; 
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Speech-to-Text (বাংলা + English)</h1>
        
        <div id="connection-status">Connecting...</div>
        
        <div class="controls">
            <button id="start">🎙️ Start Recording</button>
            <button id="stop" disabled>⏹️ Stop Recording</button>
            <button id="clear">🗑️ Clear</button>
        </div>
        
        <div id="status">Click "Start Recording" to begin</div>
        <textarea id="transcription" readonly placeholder="Transcribed text will appear here..."></textarea>
        
        <div class="debug" id="debug">Debug info will appear here...</div>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let stream, audioContext, processor, isRecording = false;
        
        function debugLog(message) {
            const debug = document.getElementById('debug');
            const timestamp = new Date().toLocaleTimeString();
            debug.textContent = `[${timestamp}] ${message}\\n` + debug.textContent;
        }

        socket.on('connect', () => {
            debugLog('Connected to server');
            document.getElementById('connection-status').textContent = '🟢 Connected';
            document.getElementById('connection-status').className = 'connected';
        });

        socket.on('disconnect', () => {
            debugLog('Disconnected from server');
            document.getElementById('connection-status').textContent = '🔴 Disconnected';
            document.getElementById('connection-status').className = '';
        });

        document.getElementById("start").onclick = async () => {
            if (isRecording) return;
            try {
                debugLog('Requesting microphone access...');
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                
                debugLog('Microphone access granted');
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                let sampleCount = 0;
                processor.onaudioprocess = e => {
                    if (!isRecording) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const samples = new Int16Array(inputData.length);
                    
                    for (let i = 0; i < inputData.length; i++) {
                        samples[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    
                    sampleCount += samples.length;
                    if (sampleCount % 16000 === 0) {  // Every second
                        debugLog(`Sent ${sampleCount} samples so far`);
                    }
                    
                    socket.emit("audio_stream", { audio: Array.from(samples) });
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                document.getElementById("start").disabled = true;
                document.getElementById("stop").disabled = false;
                socket.emit("start_streaming");
                debugLog('Recording started');
                
            } catch (err) {
                debugLog('Error: ' + err.message);
                document.getElementById("status").innerText = "❌ Error: " + err.message;
            }
        };

        document.getElementById("stop").onclick = () => {
            if (isRecording) {
                isRecording = false;
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (stream) stream.getTracks().forEach(track => track.stop());
                socket.emit("stop_streaming");
                document.getElementById("start").disabled = false;
                document.getElementById("stop").disabled = true;
                debugLog('Recording stopped');
            }
        };

        document.getElementById("clear").onclick = () => {
            document.getElementById("transcription").value = "";
            document.getElementById("debug").textContent = "";
        };

        socket.on("transcription", data => {
            debugLog('Received transcription: ' + data.text);
            const textarea = document.getElementById("transcription");
            textarea.value += data.text + "\\n";
            textarea.scrollTop = textarea.scrollHeight;
        });

        socket.on("status", data => {
            document.getElementById("status").innerText = data.status;
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on("start_streaming")
def handle_start_streaming():
    global processing_thread, stop_event, last_transcription
    logger.info("Starting streaming...")
    stop_event.clear()
    last_transcription = ""
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_audio_simple, daemon=True)
        processing_thread.start()
    
    emit("status", {"status": "🎙️ Recording... Speak in Bangla or English"})

@socketio.on("audio_stream")
def handle_audio_stream(data):
    if stop_event.is_set():
        return
        
    try:
        audio_samples = np.array(data["audio"], dtype=np.int16)
        audio_bytes = audio_samples.tobytes()
        audio_buffer.add_audio(audio_bytes)
    except Exception as e:
        logger.error(f"Error handling audio stream: {e}")

@socketio.on("stop_streaming")
def handle_stop_streaming():
    logger.info("Stopping streaming...")
    stop_event.set()
    emit("status", {"status": "⏹️ Recording stopped"})

if __name__ == "__main__":
    logger.info("Starting Speech-to-Text server...")
    socketio.run(app, host="0.0.0.0", port=7860, debug=False, allow_unsafe_werkzeug=True)