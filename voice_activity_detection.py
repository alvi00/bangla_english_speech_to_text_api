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
import webrtcvad

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize Speech2Text
stt = Speech2Text(
    model_size_or_path="tiny",
    device="cuda",
    compute_type="int8",
    cpu_threads=0,
    num_workers=1
)

# Voice Activity Detection setup
vad = webrtcvad.Vad()
vad.set_mode(2)  # Aggressiveness level (0-3, 3 is most aggressive)

class VADBuffer:
    def __init__(self, sample_rate=16000, frame_duration=30):  # 30ms frames
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.buffer = []
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = 10  # Minimum frames to consider as speech
        self.max_silence_frames = 15  # Max silence frames before ending speech
        
    def add_audio(self, audio_data):
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self.buffer.extend(audio_array)
        
        # Process complete frames
        while len(self.buffer) >= self.frame_size:
            frame = np.array(self.buffer[:self.frame_size], dtype=np.int16)
            self.buffer = self.buffer[self.frame_size:]
            
            # Check if frame contains speech
            frame_bytes = frame.tobytes()
            try:
                is_speech = vad.is_speech(frame_bytes, self.sample_rate)
                self.process_frame(frame, is_speech)
            except:
                # If VAD fails, assume it's speech
                self.process_frame(frame, True)
    
    def process_frame(self, frame, is_speech):
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.speech_buffer.extend(frame)
            
            if not self.is_speech_active and self.speech_frames >= self.min_speech_frames:
                self.is_speech_active = True
                print("Speech started")
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            
            if self.is_speech_active:
                self.speech_buffer.extend(frame)  # Include some silence
                
                if self.silence_frames >= self.max_silence_frames:
                    # End of speech detected
                    if len(self.speech_buffer) > 0:
                        speech_audio = np.array(self.speech_buffer, dtype=np.int16)
                        self.trigger_transcription(speech_audio.tobytes())
                    
                    self.reset_speech_buffer()
    
    def trigger_transcription(self, audio_data):
        # Add to transcription queue
        transcription_queue.put(audio_data)
        print(f"Triggered transcription for {len(audio_data)} bytes")
    
    def reset_speech_buffer(self):
        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_frames = 0
        self.speech_frames = 0
        print("Speech ended")

# Global variables
vad_buffer = VADBuffer()
transcription_queue = queue.Queue()
stop_event = threading.Event()
processing_thread = None

def process_transcriptions():
    print("Transcription processing thread started")
    
    while not stop_event.is_set():
        try:
            audio_data = transcription_queue.get(timeout=1.0)
            print(f"Processing transcription for {len(audio_data)} bytes")
            
            # Create audio segment
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=16000,
                channels=1
            )
            
            # Only process if audio is long enough
            if len(audio_segment) < 500:  # Less than 0.5 seconds
                continue
            
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Transcribe
            segments = stt.recognize(wav_io, return_segments=True)
            
            if segments:
                text = " ".join([segment.text.strip() for segment in segments])
                if text.strip():
                    timestamp = time.strftime('%H:%M:%S')
                    output = f"[{timestamp}] {text}"
                    socketio.emit("transcription", {
                        "text": output,
                        "status": "🎙️ Listening...",
                        "is_final": True
                    })
                    print(f"VAD Transcription: {text}")
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in transcription processing: {e}")
    
    print("Transcription processing thread stopped")

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VAD-Based Bangla Speech to Text</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
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
        }
        #transcription { 
            width: 100%; 
            height: 400px; 
            background: rgba(255,255,255,0.95);
            color: #333;
            border: none;
            border-radius: 15px;
            padding: 20px;
            font-size: 16px;
            line-height: 1.6;
            font-family: 'Courier New', monospace;
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
        #stop {
            background: linear-gradient(45deg, #f44336, #da190b);
            color: white;
        }
        .vad-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ccc;
            margin: 0 10px;
            transition: background 0.3s ease;
        }
        .vad-indicator.active {
            background: #4CAF50;
            box-shadow: 0 0 20px #4CAF50;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .info {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 VAD-Based Bangla Speech to Text</h1>
        
        <div class="info">
            <p>🚀 Advanced Voice Activity Detection - Automatically detects when you speak!</p>
            <p>Speak naturally, pauses will trigger transcription</p>
        </div>
        
        <div class="controls">
            <button id="start">🎙️ Start Listening</button>
            <button id="stop" disabled>⏹️ Stop Listening</button>
            <span class="vad-indicator" id="vad-indicator"></span>
            <span id="vad-status">Ready</span>
        </div>
        
        <textarea id="transcription" readonly placeholder="VAD-detected speech will appear here automatically..."></textarea>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let isListening = false;
        let audioContext;
        let processor;
        let stream;

        socket.on('connect', function() {
            console.log('Connected to VAD server');
        });

        document.getElementById("start").onclick = async () => {
            if (isListening) return;
            
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: false,  // Disable to let VAD work better
                        autoGainControl: true
                    } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });
                
                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(1024, 1, 1);
                
                processor.onaudioprocess = function(e) {
                    if (!isListening) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const samples = new Int16Array(inputData.length);
                    
                    for (let i = 0; i < inputData.length; i++) {
                        samples[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    
                    socket.emit("vad_audio", { 
                        audio: Array.from(samples)
                    });
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isListening = true;
                document.getElementById("start").disabled = true;
                document.getElementById("stop").disabled = false;
                document.getElementById("vad-status").textContent = "Listening...";
                
                socket.emit("start_vad");
                
            } catch (err) {
                console.error("Error:", err);
                alert("Error accessing microphone: " + err.message);
            }
        };

        document.getElementById("stop").onclick = () => {
            if (isListening) {
                isListening = false;
                
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (stream) stream.getTracks().forEach(track => track.stop());
                
                socket.emit("stop_vad");
                
                document.getElementById("start").disabled = false;
                document.getElementById("stop").disabled = true;
                document.getElementById("vad-status").textContent = "Stopped";
                document.getElementById("vad-indicator").classList.remove("active");
            }
        };

        socket.on("vad_status", data => {
            const indicator = document.getElementById("vad-indicator");
            const status = document.getElementById("vad-status");
            
            if (data.speech_detected) {
                indicator.classList.add("active");
                status.textContent = "🗣️ Speech detected";
            } else {
                indicator.classList.remove("active");
                status.textContent = "🤫 Listening...";
            }
        });

        socket.on("transcription", data => {
            const textarea = document.getElementById("transcription");
            textarea.value += data.text + "\\n";
            textarea.scrollTop = textarea.scrollHeight;
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on("start_vad")
def handle_start_vad():
    global processing_thread, stop_event
    print("Start VAD listening")
    
    stop_event.clear()
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_transcriptions, daemon=True)
        processing_thread.start()

@socketio.on("vad_audio")
def handle_vad_audio(data):
    if stop_event.is_set():
        return
    
    audio_samples = np.array(data["audio"], dtype=np.int16)
    audio_bytes = audio_samples.tobytes()
    
    # Add to VAD buffer
    vad_buffer.add_audio(audio_bytes)
    
    # Send VAD status to client
    emit("vad_status", {"speech_detected": vad_buffer.is_speech_active})

@socketio.on("stop_vad")
def handle_stop_vad():
    print("Stop VAD listening")
    stop_event.set()

if __name__ == "__main__":
    # Install webrtcvad if not installed
    try:
        import webrtcvad
    except ImportError:
        print("Installing webrtcvad...")
        os.system("pip install webrtcvad")
        import webrtcvad
    
    print("Starting VAD-based Flask-SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=7860, debug=True, allow_unsafe_werkzeug=True)