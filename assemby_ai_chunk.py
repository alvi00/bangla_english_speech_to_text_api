import asyncio
import os
import threading
import time
import logging
import sys
import numpy as np
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import assemblyai as aai
import wave
import array

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid.uuid4())
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', transports=['polling'])

# Load environment variables
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY not found in .env file")
    sys.exit(1)

# Initialize AssemblyAI client
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Thread pool for transcription
executor = ThreadPoolExecutor(max_workers=5)

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio amplitude to improve transcription clarity."""
    try:
        audio_array = np.array(audio_data, dtype=np.float32)
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0:
            audio_array = audio_array / max_amplitude * 0.9
        return (audio_array * 32768).astype(np.int16)
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        return audio_data

def save_to_wav(audio_data: list, filename: str):
    """Save audio data as a WAV file."""
    audio_data = normalize_audio(audio_data)
    audio_bytes = array.array('h', audio_data).tobytes()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)

def transcribe_wav(filename: str) -> dict:
    """Transcribe a WAV file using AssemblyAI with nano model for Bangla."""
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            language_code="bn",  # Bangla
            speech_model=aai.SpeechModel.nano  # Use nano model for Bangla support
        )
        transcription = transcriber.transcribe(filename, config)
        if transcription.error:
            raise Exception(transcription.error)
        logger.info(f"Raw transcription: {transcription.text}")
        return {'text': transcription.text}
    except Exception as e:
        logger.error(f"AssemblyAI transcription error: {str(e)}")
        return {'text': '', 'error': str(e)}

class RealtimeTranscriber:
    """Manages chunked real-time transcription."""
    
    def __init__(self):
        self.event_loop = None
        self.loop_thread = None
        
    def start_event_loop(self):
        """Start asyncio event loop in a separate thread."""
        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
            
        if not self.loop_thread or not self.loop_thread.is_alive():
            self.loop_thread = threading.Thread(target=run_loop, daemon=True)
            self.loop_thread.start()
            time.sleep(0.2)
            
    def run_async(self, coro):
        """Run async coroutine in event loop."""
        if not self.event_loop:
            self.start_event_loop()
            
        try:
            return asyncio.run_coroutine_threadsafe(coro, self.event_loop)
        except Exception as e:
            logger.error(f"Error running async task: {str(e)}")
            return None

ENHANCED_HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Speech-to-Text</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 800px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
        }
        #start-realtime {
            background: linear-gradient(45deg, #4CAF50, #45a049);
        }
        #stop {
            background: linear-gradient(45deg, #f44336, #da190b);
        }
        #clear {
            background: linear-gradient(45deg, #FF9800, #F57C00);
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .mode-panel {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 2px solid transparent;
        }
        .mode-panel.active {
            border-color: #4CAF50;
        }
        .status-text {
            font-size: 0.9em;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .output-panel {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            color: #333;
            margin-top: 20px;
        }
        .transcription-area {
            width: 100%;
            height: 300px;
            border: none;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            resize: vertical;
            background: #f9f9f9;
            unicode-bidi: embed;
            direction: ltr;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #FFD700;
        }
        .debug-info {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            font-family: monospace;
            font-size: 12px;
        }
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: center;
            }
            button {
                width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Bangla Speech-to-Text</h1>
        
        <div class="controls">
            <button id="start-realtime">🚀 Start Realtime STT</button>
            <button id="stop" disabled>⏹️ Stop Recording</button>
            <button id="clear">🗑️ Clear</button>
        </div>
        
        <div class="mode-panel" id="realtime-panel">
            <div class="status-text" id="realtime-status-text">Ready to connect...</div>
        </div>
        
        <div class="output-panel">
            <div>Realtime Transcription</div>
            <textarea class="transcription-area" id="realtime-output" readonly
                placeholder="Transcriptions will appear here..."></textarea>
        </div>
        
        <div class="stats">
            <div>
                <div class="stat-value" id="realtime-words">0</div>
                <div>Words</div>
            </div>
            <div>
                <div class="stat-value" id="session-time">00:00</div>
                <div>Session Time</div>
            </div>
        </div>
        
        <div class="debug-info" id="debug-info">Debug: Waiting for connections...</div>
    </div>
    
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
    <script>
        const socket = io({ transports: ['polling'] });
        let mediaRecorder, stream, audioContext, processor;
        let isRecording = false;
        let sessionStartTime;
        let audioBuffer = [];
        const samplesPer10Sec = 160000;

        function debugLog(message) {
            console.log(message);
            document.getElementById('debug-info').textContent = `Debug: ${message}`;
        }

        socket.on('connect', () => debugLog('Socket.IO connected'));
        socket.on('disconnect', () => debugLog('Socket.IO disconnected'));
        socket.on('connect_error', (err) => debugLog(`Socket.IO error: ${err.message}`));

        function updateSessionTimer() {
            if (!sessionStartTime) return;
            const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('session-time').textContent =
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        setInterval(updateSessionTimer, 1000);

        document.addEventListener('DOMContentLoaded', () => {
            const elements = {
                startRealtime: document.getElementById('start-realtime'),
                stop: document.getElementById('stop'),
                clear: document.getElementById('clear')
            };

            elements.startRealtime.addEventListener('click', startRecording);
            elements.stop.addEventListener('click', stopRecording);
            elements.clear.addEventListener('click', clearTranscriptions);
        });

        async function startRecording() {
            if (isRecording) {
                debugLog('Already recording');
                return;
            }

            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });

                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });

                const source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const samples = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        samples[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    audioBuffer.push(...samples);
                    while (audioBuffer.length >= samplesPer10Sec) {
                        const chunk = audioBuffer.slice(0, samplesPer10Sec);
                        socket.emit('audio_chunk', { audio: chunk });
                        audioBuffer = audioBuffer.slice(samplesPer10Sec);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                sessionStartTime = Date.now();

                document.getElementById('start-realtime').disabled = true;
                document.getElementById('stop').disabled = false;

                document.getElementById('realtime-panel').classList.add('active');
                document.getElementById('realtime-status-text').textContent = 'Recording...';

                socket.emit('start_realtime_streaming');
                debugLog('Chunked recording started');
            } catch (err) {
                debugLog(`Error starting recording: ${err.message}`);
                alert(`Error: ${err.message}`);
            }
        }

        function stopRecording() {
            if (!isRecording) return;

            isRecording = false;
            sessionStartTime = null;

            try {
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (stream) stream.getTracks().forEach(track => track.stop());

                if (audioBuffer.length > 0) {
                    socket.emit('audio_chunk', { audio: audioBuffer });
                    audioBuffer = [];
                }

                socket.emit('stop_streaming');

                document.getElementById('start-realtime').disabled = false;
                document.getElementById('stop').disabled = true;

                document.getElementById('realtime-panel').classList.remove('active');
                document.getElementById('realtime-status-text').textContent = 'OK';

                debugLog('Recording stopped');
            } catch (err) {
                debugLog(`Error stopping recording: ${err.message}`);
            }
        }

        function clearTranscriptions() {
            document.getElementById('realtime-output').value = '';
            document.getElementById('realtime-words').textContent = '0';
            debugLog('Transcriptions cleared');
        }

        socket.on('realtime_transcription', (data) => {
            const textarea = document.getElementById('realtime-output');
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            const text = data.error ? `Error: ${data.error}` : data.text;
            textarea.value += `[${timestamp}] ${text}\n`;
            textarea.scrollTop = textarea.scrollHeight;

            if (!data.error) {
                const words = textarea.value.split(/\s+/).filter(word => word.length > 0);
                document.getElementById('realtime-words').textContent = words.length;
            }
        });

        socket.on('realtime_status', (data) => {
            document.getElementById('realtime-status-text').textContent = data.status || data.error;
            debugLog(`Realtime data: ${data.status || data.error}`);
        });

        debugLog('Page loaded');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(ENHANCED_HTML_TEMPLATE)

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    """Handle Chrome DevTools request to avoid 404 errors."""
    return jsonify({}), 200

@socketio.on('start_realtime_streaming')
def handle_start_realtime_streaming():
    logger.info("Starting chunked Realtime STT")
    try:
        socketio.emit('realtime_status', {'status': '🔄 Starting chunked STT...'})
    except Exception as e:
        logger.error(f"Error starting chunked STT: {str(e)}")
        socketio.emit('realtime_status', {'error': f"❌ Failed to start: {str(e)}"})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    audio_data = data.get('audio', [])
    wav_filename = f"temp_{uuid.uuid4()}.wav"
    try:
        save_to_wav(audio_data, wav_filename)
        future = executor.submit(transcribe_wav, wav_filename)
        def callback(fut):
            try:
                transcription_result = fut.result()
                socketio.emit('realtime_transcription', {
                    'text': transcription_result['text'],
                    'source': 'assemblyai',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                socketio.emit('realtime_transcription', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            finally:
                if os.path.exists(wav_filename):
                    os.remove(wav_filename)
        future.add_done_callback(callback)
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        socketio.emit('realtime_status', {'error': f"❌ Error processing chunk: {str(e)}"})

@socketio.on('stop_streaming')
def handle_stop_streaming():
    logger.info("Stopping chunked Realtime STT")
    try:
        socketio.emit('realtime_status', {'status': '🔴 Chunked STT stopped'})
    except Exception as e:
        logger.error(f"Error stopping chunked STT: {str(e)}")

if __name__ == '__main__':
    logger.info("Starting Bangla Speech-to-Text server with chunked STT...")
    socketio.run(app, host='0.0.0.0', port=7860, debug=False, allow_unsafe_werkzeug=True)