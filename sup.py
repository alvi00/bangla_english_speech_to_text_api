from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from banglaspeech2text import Speech2Text
import torch
import io
import time
import threading
import queue
from pydub import AudioSegment
import os
import numpy as np
from collections import deque

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize Speech2Text with 'large' model
stt = Speech2Text(
    model_size_or_path="large",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)

# Audio buffer for streaming
class AudioBuffer:
    def __init__(self, max_duration=30.0, sample_rate=16000):
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_data):
        with self.lock:
            # Convert audio to numpy array and add to buffer
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.buffer.extend(audio_array)
    
    def get_audio_segment(self, duration=5.0):
        with self.lock:
            samples_needed = int(duration * self.sample_rate)
            if len(self.buffer) < samples_needed:
                return None
            
            # Get the last N seconds of audio
            audio_segment = np.array(list(self.buffer)[-samples_needed:])
            return audio_segment.astype(np.int16).tobytes()

# Global variables
audio_buffer = AudioBuffer()
stop_event = threading.Event()
transcription_lock = threading.Lock()
processing_thread = None
last_transcription = ""

def process_audio_streaming():
    global last_transcription
    print("Streaming audio processing thread started")
    
    while not stop_event.is_set():
        try:
            # Get 5 seconds of audio
            audio_data = audio_buffer.get_audio_segment(duration=5.0)
            
            if audio_data is None:
                time.sleep(0.5)
                continue
            
            print(f"Processing {len(audio_data)} bytes of audio")
            
            # Create audio segment from raw PCM data
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=16000,
                channels=1
            )
            
            # Convert to WAV format
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Transcribe
            with transcription_lock:
                segments = stt.recognize(wav_io, return_segments=True)
            
            if segments:
                current_text = " ".join([segment.text.strip() for segment in segments])
                
                # Only emit if text is different from last transcription
                if current_text != last_transcription and current_text.strip():
                    timestamp = time.strftime('%H:%M:%S')
                    output = f"[{timestamp}] {current_text}"
                    socketio.emit("transcription", {
                        "text": output, 
                        "status": "Recording... Speak in Bangla.",
                        "is_final": False
                    })
                    print(f"Streaming transcription: {current_text}")
                    last_transcription = current_text
            
            time.sleep(1.0)  # Process every 1 second
            
        except Exception as e:
            print(f"Error in streaming process: {e}")
            time.sleep(1.0)
    
    print("Streaming audio processing thread stopped")

# HTML template with real-time features
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Bangla Speech to Text</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        #transcription { 
            width: 100%; 
            height: 300px; 
            background: rgba(255,255,255,0.9);
            color: #333;
            border: none;
            border-radius: 10px;
            padding: 15px;
            font-size: 16px;
            line-height: 1.5;
        }
        #status { 
            margin: 15px 0; 
            font-weight: bold; 
            padding: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            text-align: center;
        }
        button { 
            padding: 12px 24px; 
            margin: 8px; 
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        #start {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        #start:hover { transform: translateY(-2px); }
        #start:disabled { 
            background: #ccc; 
            cursor: not-allowed;
            transform: none;
        }
        #stop {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        #stop:hover { transform: translateY(-2px); }
        #stop:disabled { 
            background: #ccc; 
            cursor: not-allowed;
            transform: none;
        }
        #clear {
            background: linear-gradient(45deg, #ff9800, #f57c00);
            color: white;
        }
        #clear:hover { transform: translateY(-2px); }
        #connection-status { 
            margin-bottom: 15px;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            background: rgba(244, 67, 54, 0.8);
        }
        #connection-status.connected { 
            background: rgba(76, 175, 80, 0.8);
        }
        .recording-indicator {
            display: none;
            color: #ff4444;
            font-weight: bold;
            animation: blink 1s infinite;
        }
        .recording-indicator.active { display: inline; }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Real-time Bangla Speech to Text</h1>
        <div id="connection-status">Connecting...</div>
        
        <div class="controls">
            <button id="start">🎙️ Start Recording</button>
            <button id="stop" disabled>⏹️ Stop Recording</button>
            <button id="clear">🗑️ Clear Text</button>
            <span class="recording-indicator" id="recording">● REC</span>
        </div>
        
        <div id="status">Click "Start Recording" to begin real-time transcription</div>
        <textarea id="transcription" readonly placeholder="Real-time transcribed text will appear here..."></textarea>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let mediaRecorder;
        let stream;
        let isRecording = false;
        let audioContext;
        let processor;

        // Connection status
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('connection-status').textContent = '🟢 Connected to server';
            document.getElementById('connection-status').className = 'connected';
        });

        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('connection-status').textContent = '🔴 Disconnected from server';
            document.getElementById('connection-status').className = '';
        });

        document.getElementById("start").onclick = async () => {
            if (isRecording) return;
            
            try {
                console.log('Requesting microphone access...');
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                console.log('Microphone access granted');
                
                // Setup Web Audio API for real-time processing
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });
                
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = function(e) {
                    if (!isRecording) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const samples = new Int16Array(inputData.length);
                    
                    // Convert float32 to int16
                    for (let i = 0; i < inputData.length; i++) {
                        samples[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    
                    // Send audio data
                    socket.emit("audio_stream", { 
                        audio: Array.from(samples)
                    });
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                
                document.getElementById("start").disabled = true;
                document.getElementById("stop").disabled = false;
                document.getElementById("recording").classList.add("active");
                
                socket.emit("start_streaming");
                console.log('Real-time recording started');
                
            } catch (err) {
                console.error("Error accessing microphone:", err);
                document.getElementById("status").innerText = "Error: " + err.message;
            }
        };

        document.getElementById("stop").onclick = () => {
            if (isRecording) {
                console.log('Stopping recording...');
                
                isRecording = false;
                
                if (processor) {
                    processor.disconnect();
                }
                if (audioContext) {
                    audioContext.close();
                }
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                
                socket.emit("stop_streaming");
                
                document.getElementById("start").disabled = false;
                document.getElementById("stop").disabled = true;
                document.getElementById("recording").classList.remove("active");
            }
        };

        document.getElementById("clear").onclick = () => {
            document.getElementById("transcription").value = "";
        };

        socket.on("status", data => {
            console.log('Status update:', data.status);
            document.getElementById("status").innerText = data.status;
        });

        socket.on("transcription", data => {
            console.log('Transcription received:', data.text);
            const textarea = document.getElementById("transcription");
            
            if (data.is_final) {
                textarea.value += data.text + "\\n";
            } else {
                // For streaming, replace the last line or add new line
                const lines = textarea.value.split('\\n');
                if (lines.length > 0 && lines[lines.length - 1].startsWith('[' + data.text.substring(1, 9))) {
                    lines[lines.length - 1] = data.text;
                } else {
                    lines.push(data.text);
                }
                textarea.value = lines.join('\\n');
            }
            
            textarea.scrollTop = textarea.scrollHeight;
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
    print("Start streaming request received")
    
    # Reset everything
    stop_event.clear()
    last_transcription = ""
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_audio_streaming, daemon=True)
        processing_thread.start()
        print("Started streaming audio processing thread")
    
    emit("status", {"status": "🎙️ Real-time recording active... Speak in Bangla"})

@socketio.on("audio_stream")
def handle_audio_stream(data):
    if stop_event.is_set():
        return
    
    # Convert Int16Array to bytes
    audio_samples = np.array(data["audio"], dtype=np.int16)
    audio_bytes = audio_samples.tobytes()
    
    # Add to circular buffer
    audio_buffer.add_audio(audio_bytes)

@socketio.on("stop_streaming")
def handle_stop_streaming():
    print("Stop streaming request received")
    stop_event.set()
    emit("status", {"status": "⏹️ Recording stopped"})

@socketio.on('connect')
def handle_connect():
    print('Client connected for streaming')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from streaming')

if __name__ == "__main__":
    print("Starting Real-time Flask-SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=7860, debug=True, allow_unsafe_werkzeug=True)