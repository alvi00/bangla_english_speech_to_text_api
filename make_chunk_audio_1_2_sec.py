from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from banglaspeech2text import Speech2Text
import torch
import io
import time
import threading
import queue
from pydub import AudioSegment
import numpy as np
from collections import deque
import logging
import webrtcvad
from scipy import signal
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# Check GPU availability
logger.info(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize Speech2Text with optimized settings
stt = Speech2Text(
    # model_size_or_path="openai/whisper-small",  # Supports 100+ languages including Bangla/English
    model_size_or_path="large",  # Supports 100+ languages including Bangla/English
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",  # Optimized for GTX 1050
    cpu_threads=4,
    num_workers=2
)

# Voice Activity Detection
vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3

class SmartAudioBuffer:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # Sliding window for better transcription
        self.window_duration = 5.0  # 5 second sliding window
        self.window_samples = int(sample_rate * self.window_duration)
        
        self.buffer = deque(maxlen=self.window_samples)
        self.speech_chunks = deque(maxlen=20)  # Store recent speech chunks
        self.lock = threading.Lock()
        
        # Voice activity tracking
        self.speech_frames = deque(maxlen=30)  # Last 30 frames for VAD
        self.silence_duration = 0
        self.speech_duration = 0
        
    def add_audio(self, audio_data):
        with self.lock:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.buffer.extend(audio_array)
            
            # Process in chunks for VAD
            while len(self.buffer) >= self.chunk_samples:
                chunk = np.array([self.buffer.popleft() for _ in range(self.chunk_samples)])
                self._process_chunk_vad(chunk)
    
    def _process_chunk_vad(self, chunk):
        """Process chunk with Voice Activity Detection"""
        try:
            # Convert to bytes for webrtcvad
            chunk_bytes = (chunk).astype(np.int16).tobytes()
            
            # VAD requires specific frame sizes (10, 20, or 30ms)
            frame_duration = 30  # ms
            frame_samples = int(self.sample_rate * frame_duration / 1000)
            
            has_speech = False
            for i in range(0, len(chunk), frame_samples):
                frame = chunk[i:i + frame_samples]
                if len(frame) == frame_samples:
                    frame_bytes = frame.astype(np.int16).tobytes()
                    if vad.is_speech(frame_bytes, self.sample_rate):
                        has_speech = True
                        break
            
            self.speech_frames.append(has_speech)
            
            if has_speech:
                self.speech_duration += self.chunk_duration
                self.silence_duration = 0
                self.speech_chunks.append(chunk)
            else:
                self.silence_duration += self.chunk_duration
                if self.speech_duration > 0:  # Keep some silence after speech
                    self.speech_chunks.append(chunk * 0.1)  # Reduced volume silence
        
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Fallback: assume speech if VAD fails
            self.speech_chunks.append(chunk)
    
    def should_process(self):
        """Determine if we should process accumulated speech"""
        recent_speech = sum(self.speech_frames) if self.speech_frames else 0
        return (
            recent_speech >= 3 and  # At least 3 speech frames detected
            (self.silence_duration >= 1.0 or  # 1 second of silence, or
             self.speech_duration >= 4.0)     # 4 seconds of continuous speech
        )
    
    def get_speech_audio(self):
        """Get accumulated speech audio for transcription"""
        with self.lock:
            if not self.speech_chunks:
                return None
            
            # Combine speech chunks
            audio_data = np.concatenate(list(self.speech_chunks))
            
            # Clear processed chunks
            self.speech_chunks.clear()
            self.speech_duration = 0
            
            return audio_data.astype(np.int16).tobytes()

# Transcription cache to avoid duplicates
class TranscriptionCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        
    def get_hash(self, audio_data):
        """Generate hash for audio data"""
        return hashlib.md5(audio_data).hexdigest()[:16]
    
    def is_duplicate(self, audio_data, threshold=0.8):
        """Check if this audio is similar to recent transcriptions"""
        current_hash = self.get_hash(audio_data)
        current_time = time.time()
        
        # Clean old entries
        old_keys = [k for k, t in self.timestamps.items() if current_time - t > 30]
        for k in old_keys:
            self.cache.pop(k, None)
            self.timestamps.pop(k, None)
        
        # Check for similar hashes (simple approach)
        for cached_hash in self.cache.keys():
            similarity = sum(a == b for a, b in zip(current_hash, cached_hash)) / len(current_hash)
            if similarity > threshold:
                return True, self.cache[cached_hash]
        
        return False, None
    
    def add(self, audio_data, transcription):
        """Add transcription to cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=self.timestamps.get)
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key)
        
        audio_hash = self.get_hash(audio_data)
        self.cache[audio_hash] = transcription
        self.timestamps[audio_hash] = time.time()

# Global variables
audio_buffer = SmartAudioBuffer()
transcription_cache = TranscriptionCache()
stop_event = threading.Event()
processing_thread = None
last_transcription = ""
session_transcripts = deque(maxlen=50)  # Keep session history

def process_audio_intelligent():
    """Intelligent audio processing with VAD and caching"""
    global last_transcription
    logger.info("Intelligent audio processing thread started")
    
    while not stop_event.is_set():
        try:
            if not audio_buffer.should_process():
                time.sleep(0.1)
                continue
                
            audio_data = audio_buffer.get_speech_audio()
            if audio_data is None or len(audio_data) < 16000:  # Less than 1 second
                continue
            
            # Check cache first
            is_duplicate, cached_result = transcription_cache.is_duplicate(audio_data)
            if is_duplicate and cached_result:
                logger.info("Using cached transcription")
                current_text = cached_result
            else:
                # Process with Whisper
                logger.info(f"Processing {len(audio_data)} bytes of speech audio")
                
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                
                # Apply noise reduction
                audio_segment = audio_segment.normalize()
                
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                # Transcribe with language detection
                segments = stt.recognize(
                    wav_io, 
                    return_segments=True,
                    language=None,  # Auto-detect Bangla/English
                    task="transcribe"
                )
                
                if segments:
                    current_text = " ".join([segment.text.strip() for segment in segments])
                    # Cache the result
                    transcription_cache.add(audio_data, current_text)
                else:
                    current_text = ""
            
            # Only emit if significantly different from last transcription
            if current_text and current_text.strip():
                # Simple similarity check
                if not last_transcription or len(set(current_text.split()) - set(last_transcription.split())) > 1:
                    timestamp = time.strftime('%H:%M:%S')
                    output = f"[{timestamp}] {current_text}"
                    
                    session_transcripts.append(output)
                    
                    socketio.emit("transcription", {
                        "text": output,
                        "status": "🎙️ Listening... (Bangla/English)",
                        "is_final": True,
                        "confidence": "high" if not is_duplicate else "cached"
                    })
                    
                    logger.info(f"New transcription: {current_text}")
                    last_transcription = current_text
            
            time.sleep(0.2)  # Reduced processing frequency
            
        except Exception as e:
            logger.error(f"Error in intelligent processing: {e}")
            time.sleep(1.0)
    
    logger.info("Intelligent audio processing thread stopped")

# Enhanced HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Speech-to-Text (বাংলা + English)</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(15px);
            margin-top: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        #transcription { 
            width: 100%; 
            height: 350px; 
            background: rgba(255,255,255,0.95);
            color: #333;
            border: none;
            border-radius: 12px;
            padding: 20px;
            font-size: 16px;
            line-height: 1.6;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        #status { 
            margin: 20px 0; 
            font-weight: bold; 
            padding: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #4CAF50;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
            font-size: 14px;
        }
        .stat-item {
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        button { 
            padding: 15px 30px; 
            margin: 10px; 
            border: none;
            border-radius: 30px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        #start {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }
        #start:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
        #start:disabled { 
            background: #666; 
            cursor: not-allowed;
            transform: none;
        }
        #stop {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        #stop:hover { transform: translateY(-3px); }
        #stop:disabled { 
            background: #666; 
            cursor: not-allowed;
            transform: none;
        }
        #clear, #export {
            background: linear-gradient(45deg, #ff9800, #f57c00);
            color: white;
        }
        #clear:hover, #export:hover { transform: translateY(-3px); }
        #connection-status { 
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            background: rgba(244, 67, 54, 0.8);
            font-weight: bold;
        }
        #connection-status.connected { 
            background: rgba(76, 175, 80, 0.8);
        }
        .recording-indicator {
            display: none;
            color: #ff4444;
            font-weight: bold;
            animation: pulse 1.5s infinite;
            font-size: 18px;
        }
        .recording-indicator.active { display: inline-block; }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        .controls {
            text-align: center;
            margin: 25px 0;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            font-size: 2.2em;
        }
        .language-indicator {
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            margin: 0 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Smart Speech-to-Text 
            <span class="language-indicator">বাংলা</span>
            <span class="language-indicator">English</span>
        </h1>
        
        <div id="connection-status">🔄 Connecting...</div>
        
        <div class="stats">
            <div class="stat-item">
                <strong>Session Time</strong><br>
                <span id="session-time">00:00</span>
            </div>
            <div class="stat-item">
                <strong>Transcriptions</strong><br>
                <span id="transcription-count">0</span>
            </div>
            <div class="stat-item">
                <strong>Voice Activity</strong><br>
                <span id="voice-activity">🔇 Silent</span>
            </div>
        </div>
        
        <div class="controls">
            <button id="start">🎙️ Start Smart Recording</button>
            <button id="stop" disabled>⏹️ Stop Recording</button>
            <button id="clear">🗑️ Clear Text</button>
            <button id="export">💾 Export Text</button>
            <div class="recording-indicator" id="recording">🔴 RECORDING</div>
        </div>
        
        <div id="status">Click "Start Smart Recording" to begin voice detection</div>
        <textarea id="transcription" readonly placeholder="Smart transcription with voice activity detection...\nSupports: বাংলা and English mixed speech"></textarea>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let stream, audioContext, processor, isRecording = false;
        let sessionStartTime = null;
        let transcriptionCount = 0;

        // Update session timer
        function updateSessionTime() {
            if (sessionStartTime && isRecording) {
                const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('session-time').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }
        setInterval(updateSessionTime, 1000);

        socket.on('connect', () => {
            console.log('Connected to server');
            document.getElementById('connection-status').textContent = '🟢 Connected - Ready for Smart Recording';
            document.getElementById('connection-status').className = 'connected';
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            document.getElementById('connection-status').textContent = '🔴 Disconnected - Reconnecting...';
            document.getElementById('connection-status').className = '';
        });

        document.getElementById("start").onclick = async () => {
            if (isRecording) return;
            try {
                console.log('Starting smart recording...');
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = e => {
                    if (!isRecording) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const samples = new Int16Array(inputData.length);
                    
                    // Simple voice activity detection on client side
                    let volume = 0;
                    for (let i = 0; i < inputData.length; i++) {
                        samples[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                        volume += Math.abs(inputData[i]);
                    }
                    volume = volume / inputData.length;
                    
                    // Update voice activity indicator
                    const voiceActivity = document.getElementById('voice-activity');
                    if (volume > 0.01) {
                        voiceActivity.innerHTML = '🎤 Speaking';
                        voiceActivity.style.color = '#4CAF50';
                    } else {
                        voiceActivity.innerHTML = '🔇 Silent';
                        voiceActivity.style.color = '#999';
                    }
                    
                    socket.emit("audio_stream", { audio: Array.from(samples), volume: volume });
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                sessionStartTime = Date.now();
                document.getElementById("start").disabled = true;
                document.getElementById("stop").disabled = false;
                document.getElementById("recording").classList.add("active");
                socket.emit("start_streaming");
            } catch (err) {
                console.error("Error:", err);
                document.getElementById("status").innerText = "❌ Error: " + err.message;
            }
        };

        document.getElementById("stop").onclick = () => {
            if (isRecording) {
                isRecording = false;
                sessionStartTime = null;
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (stream) stream.getTracks().forEach(track => track.stop());
                socket.emit("stop_streaming");
                document.getElementById("start").disabled = false;
                document.getElementById("stop").disabled = true;
                document.getElementById("recording").classList.remove("active");
                document.getElementById('voice-activity').innerHTML = '🔇 Stopped';
            }
        };

        document.getElementById("clear").onclick = () => {
            document.getElementById("transcription").value = "";
            transcriptionCount = 0;
            document.getElementById('transcription-count').textContent = '0';
        };

        document.getElementById("export").onclick = () => {
            const text = document.getElementById("transcription").value;
            if (text.trim()) {
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `transcription_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
                a.click();
                URL.revokeObjectURL(url);
            }
        };

        socket.on("transcription", data => {
            console.log('Smart transcription:', data);
            const textarea = document.getElementById("transcription");
            textarea.value += data.text + "\\n";
            textarea.scrollTop = textarea.scrollHeight;
            
            transcriptionCount++;
            document.getElementById('transcription-count').textContent = transcriptionCount;
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
    logger.info("Starting smart streaming...")
    stop_event.clear()
    last_transcription = ""
    session_transcripts.clear()
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_audio_intelligent, daemon=True)
        processing_thread.start()
    
    emit("status", {"status": "🎙️ Smart recording active - Voice detection enabled"})

@socketio.on("audio_stream")
def handle_audio_stream(data):
    if stop_event.is_set():
        return
    
    audio_samples = np.array(data["audio"], dtype=np.int16)
    audio_bytes = audio_samples.tobytes()
    audio_buffer.add_audio(audio_bytes)

@socketio.on("stop_streaming")
def handle_stop_streaming():
    logger.info("Stopping smart streaming...")
    stop_event.set()
    emit("status", {"status": "⏹️ Smart recording stopped"})

@socketio.on("get_session_summary")
def handle_session_summary():
    """Get session transcription summary"""
    summary = "\n".join(session_transcripts)
    emit("session_summary", {"summary": summary, "count": len(session_transcripts)})

if __name__ == "__main__":
    logger.info("Starting Smart Speech-to-Text server...")
    socketio.run(app, host="0.0.0.0", port=7860, debug=False, allow_unsafe_werkzeug=True)