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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize Speech2Text with 'tiny' model
stt = Speech2Text(
    model_size_or_path="tiny",
    device="cuda",
    compute_type="int8",
    cpu_threads=0,
    num_workers=1
)

# Queue for audio chunks
audio_queue = queue.Queue()
stop_event = threading.Event()
transcription_lock = threading.Lock()
processing_thread = None

# Process audio chunks in a separate thread
def process_audio():
    print("Audio processing thread started")
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.5)  # Use timeout instead of get_nowait
            print(f"Processing audio chunk of size: {len(audio_data)} bytes")
            
            # Convert WebM audio to WAV
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                # Save for debugging
                with open("test_chunk.wav", "wb") as f:
                    f.write(wav_io.getvalue())
                print("Saved audio chunk as test_chunk.wav")
                
                # Reset wav_io position
                wav_io.seek(0)
                
                # Transcribe with thread safety
                with transcription_lock:
                    segments = stt.recognize(wav_io, return_segments=True)
                
                if not segments:
                    print("No segments detected in audio")
                    socketio.emit("transcription", {"text": "[No speech detected]", "status": "Recording... Speak in Bangla."})
                else:
                    output = "\n".join([f"[{time.strftime('%H:%M:%S')} | {segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])
                    socketio.emit("transcription", {"text": output, "status": "Recording... Speak in Bangla."})
                    print(f"Transcribed: {output}")
                    
            except Exception as audio_error:
                print(f"Audio processing error: {audio_error}")
                socketio.emit("transcription", {"text": f"Audio processing error: {str(audio_error)}", "status": "Error processing audio"})
                
        except queue.Empty:
            continue  # Timeout reached, continue loop
        except Exception as e:
            print(f"Error in process_audio: {e}")
            socketio.emit("transcription", {"text": f"Error: {str(e)}", "status": f"Error: {str(e)}"})
    
    print("Audio processing thread stopped")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bangla Speech to Text (Live, GPU)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #transcription { width: 100%; height: 200px; }
        #status { margin-top: 10px; font-weight: bold; }
        button { padding: 10px 20px; margin: 5px; }
        #connection-status { color: red; margin-bottom: 10px; }
        #connection-status.connected { color: green; }
    </style>
</head>
<body>
    <h1>Bangla Speech to Text (Live, GPU)</h1>
    <div id="connection-status">Connecting...</div>
    <p>Click "Start Recording" to begin live Bangla speech transcription. Uses the 'tiny' model for testing.</p>
    <button id="start">Start Recording</button>
    <button id="stop" disabled>Stop Recording</button>
    <div id="status">Click "Start Recording" to begin.</div>
    <textarea id="transcription" readonly placeholder="Transcribed text will appear here..."></textarea>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let mediaRecorder;
        let stream;
        let isRecording = false;

        // Connection status
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('connection-status').textContent = 'Connected to server';
            document.getElementById('connection-status').className = 'connected';
        });

        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('connection-status').textContent = 'Disconnected from server';
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
                        noiseSuppression: true
                    } 
                });
                
                console.log('Microphone access granted');
                
                // Check if webm is supported, fallback to other formats
                let mimeType = 'audio/webm; codecs=opus';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = 'audio/mp4';
                        if (!MediaRecorder.isTypeSupported(mimeType)) {
                            mimeType = '';
                        }
                    }
                }
                
                console.log('Using MIME type:', mimeType);
                
                mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
                
                mediaRecorder.ondataavailable = e => {
                    console.log('Audio data available, size:', e.data.size);
                    if (e.data.size > 0) {
                        e.data.arrayBuffer().then(buffer => {
                            console.log('Sending audio buffer of size:', buffer.byteLength);
                            socket.emit("audio", { audio: Array.from(new Uint8Array(buffer)) });
                        });
                    }
                };
                
                mediaRecorder.onerror = e => {
                    console.error('MediaRecorder error:', e);
                };
                
                mediaRecorder.start(2000); // Send audio every 2 seconds
                isRecording = true;
                
                document.getElementById("start").disabled = true;
                document.getElementById("stop").disabled = false;
                
                socket.emit("start_recording");
                console.log('Recording started');
                
            } catch (err) {
                console.error("Error accessing microphone:", err);
                document.getElementById("status").innerText = "Error: " + err.message;
            }
        };

        document.getElementById("stop").onclick = () => {
            if (mediaRecorder && isRecording) {
                console.log('Stopping recording...');
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
                socket.emit("stop");
                isRecording = false;
                
                document.getElementById("start").disabled = false;
                document.getElementById("stop").disabled = true;
            }
        };

        socket.on("status", data => {
            console.log('Status update:', data.status);
            document.getElementById("status").innerText = data.status;
        });

        socket.on("transcription", data => {
            console.log('Transcription received:', data.text);
            const textarea = document.getElementById("transcription");
            textarea.value += data.text + "\\n";
            textarea.scrollTop = textarea.scrollHeight;
        });
        
        // Test socket connection
        socket.emit("test", {message: "Client connected"});
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on("test")
def handle_test(data):
    print(f"Test message received: {data}")
    emit("status", {"status": "Server connection confirmed"})

@socketio.on("start_recording")
def handle_start_recording():
    global processing_thread, stop_event
    print("Start recording request received")
    
    # Reset stop event and start processing thread
    stop_event.clear()
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_audio, daemon=True)
        processing_thread.start()
        print("Started new audio processing thread")
    
    emit("status", {"status": "Recording started... Speak in Bangla."})

@socketio.on("audio")
def handle_audio(data):
    print(f"Received audio data, type: {type(data['audio'])}, length: {len(data['audio'])}")
    
    if stop_event.is_set():
        print("Stop event is set, ignoring audio data")
        return
    
    # Convert list back to bytes
    audio_bytes = bytes(data["audio"])
    print(f"Converted to bytes, size: {len(audio_bytes)}")
    
    audio_queue.put(audio_bytes)
    emit("status", {"status": "Processing audio... Speak in Bangla."})

@socketio.on("stop")
def handle_stop():
    print("Stop recording request received")
    stop_event.set()
    emit("status", {"status": "Recording stopped."})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    print("Starting Flask-SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=7860, debug=True, allow_unsafe_werkzeug=True)