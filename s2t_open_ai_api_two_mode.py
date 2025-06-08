import asyncio
import os
import json
import base64
import threading
import time
import logging
import sys
from typing import List, Optional, Any
import numpy as np
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import websockets
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid.uuid4())
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file")
    sys.exit(1)

# Initialize OpenAI client for Whisper
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAIRealtimeTranscriber:
    """Handles real-time transcription using OpenAI Realtime API."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.websocket: Optional[Any] = None  # Fix Pylance error
        self.is_connected = False
        self.session_id: Optional[str] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API."""
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            try:
                self.websocket = await websockets.connect(url, extra_headers=headers)
            except TypeError:
                logger.warning("Modern websockets.connect failed; attempting legacy connection")
                try:
                    from websockets.legacy.client import Connect
                    self.websocket = await Connect(url, extra_headers=headers).__aenter__()
                except Exception as e:
                    raise Exception(f"Legacy connection failed: {str(e)}")
            
            # Configure session with automatic language detection
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "voice": "alloy",
                    "instructions": (
                        "You are a helpful assistant. Transcribe audio accurately in the original "
                        "language (Bengali/Bangla or English) with automatic language detection. "
                        "Use the appropriate script for each language."
                    ),
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200
                    },
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    }
                }
            }
            
            await self.websocket.send(json.dumps(session_config))
            self.is_connected = True
            logger.info("Connected to OpenAI Realtime API")
            socketio.emit('realtime_status', {'status': '✅ Connected to OpenAI Realtime API'})
            
            asyncio.create_task(self.listen_for_responses())
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {str(e)}")
            self.is_connected = False
            socketio.emit('realtime_status', {'error': f"❌ Connection failed: {str(e)[:50]}..."})
            
    async def listen_for_responses(self):
        """Listen for responses from OpenAI Realtime API."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_response(data)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from OpenAI")
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI Realtime API connection closed")
            self.is_connected = False
            socketio.emit('realtime_status', {'status': '🔴 Connection closed'})
        except Exception as e:
            logger.error(f"Error listening to OpenAI Realtime API: {str(e)}")
            self.is_connected = False
            
    async def _handle_response(self, data: dict):
        """Process individual responses from OpenAI Realtime API."""
        event_type = data.get('type')
        
        if event_type == 'session.created':
            self.session_id = data.get('session', {}).get('id')
            logger.info(f"Session created: {self.session_id}")
            socketio.emit('realtime_status', {'status': '✅ Session created'})
            
        elif event_type == 'input_audio_buffer.speech_started':
            logger.info("Speech started detected")
            socketio.emit('realtime_status', {'status': '🎤 Speech detected...'})
            
        elif event_type == 'input_audio_buffer.speech_stopped':
            logger.info("Speech stopped detected")
            socketio.emit('realtime_status', {'status': '⏸️ Processing speech...'})
            
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            transcript = data.get('transcript', '')
            if transcript.strip():
                logger.info(f"Transcription received: {transcript}")
                # Attempt to detect language based on script
                is_bangla = any(0x0980 <= ord(c) <= 0x09FF for c in transcript)
                language = 'Bengali' if is_bangla else 'English'
                socketio.emit('realtime_transcription', {
                    'text': transcript,
                    'source': 'openai_realtime',
                    'language': language,
                    'timestamp': datetime.now().isoformat()
                })
                socketio.emit('realtime_status', {'status': f'✨ Transcription received ({language})'})
                
        elif event_type == 'error':
            error_msg = data.get('error', {}).get('message', 'Unknown error')
            logger.error(f"OpenAI API error: {error_msg}")
            socketio.emit('realtime_status', {'error': f"❌ Error: {error_msg}"})
            
    async def send_audio(self, audio_data: np.ndarray):
        """Send audio data to OpenAI Realtime API."""
        if not self.is_connected or not self.websocket:
            logger.warning("Cannot send audio: Not connected")
            return
            
        try:
            audio_array = np.array(audio_data, dtype=np.int16)
            audio_base64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')
            
            message = {
                'type': 'input_audio_buffer.append',
                'audio': audio_base64
            }
            
            await self.websocket.send(json.dumps(message))
            logger.debug("Audio chunk sent to OpenAI")
            
        except Exception as e:
            logger.error(f"Error sending audio: {str(e)}")
            
    async def commit_audio(self):
        """Commit audio buffer for processing."""
        if not self.is_connected or not self.websocket:
            return
            
        try:
            message = {'type': 'input_audio_buffer.commit'}
            await self.websocket.send(json.dumps(message))
            logger.info("Audio buffer committed")
        except Exception as e:
            logger.error(f"Error committing audio: {str(e)}")
            
    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected from OpenAI Realtime API")
            except Exception as e:
                logger.error(f"Error during disconnect: {str(e)}")
        self.is_connected = False

class WhisperTranscriber:
    """Handles traditional transcription using OpenAI Whisper."""
    
    def __init__(self):
        self.client = openai_client
        self.audio_buffer: List[np.ndarray] = []
        self.buffer_lock = threading.Lock()
        
    def add_audio_chunk(self, audio_data: List[int]):
        """Add audio chunk to buffer."""
        with self.buffer_lock:
            self.audio_buffer.append(np.array(audio_data, dtype=np.int16))
            
    def transcribe(self) -> Optional[dict]:
        """Transcribe buffered audio using Whisper with auto language detection."""
        try:
            with self.buffer_lock:
                if not self.audio_buffer:
                    return None
                    
                audio_array = np.concatenate(self.audio_buffer)
                self.audio_buffer.clear()
                
            import soundfile as sf
            temp_file = 'temp_audio.wav'
            sf.write(temp_file, audio_array, samplerate=16000, format='WAV', subtype='PCM_16')
            
            with open(temp_file, 'rb') as f:
                transcription = self.client.audio.transcriptions.create(
                    model='whisper-1',
                    file=f
                    # Omit language for auto-detection
                )
                
            os.remove(temp_file)
            # Infer language based on script
            is_bangla = any(0x0980 <= ord(c) <= 0x09FF for c in transcription.text)
            language = 'Bengali' if is_bangla else 'English'
            return {'text': transcription.text, 'language': language}
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            return None

class EnhancedHybridTranscriber:
    """Coordinates real-time and traditional transcription."""
    
    def __init__(self):
        self.realtime_transcriber = OpenAIRealtimeTranscriber()
        self.whisper_transcriber = WhisperTranscriber()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        
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
            
    def start_realtime(self):
        """Start OpenAI Realtime API connection."""
        self.run_async(self.realtime_transcriber.connect())
        
    def stop_realtime(self):
        """Stop OpenAI Realtime API connection."""
        self.run_async(self.realtime_transcriber.disconnect())
        
    def add_audio_chunk(self, audio_data: List[int], mode: str):
        """Add audio chunk to appropriate transcriber."""
        if mode == 'realtime':
            if self.realtime_transcriber.is_connected:
                self.run_async(self.realtime_transcriber.send_audio(audio_data))
        else:
            self.whisper_transcriber.add_audio_chunk(audio_data)
            
    def process_speech_end(self, mode: str):
        """Process end of speech."""
        if mode == 'realtime':
            self.run_async(self.realtime_transcriber.commit_audio())
        else:
            result = self.whisper_transcriber.transcribe()
            if result:
                socketio.emit('transcription', {
                    'text': result['text'],
                    'source': 'whisper',
                    'language': result['language'],
                    'timestamp': datetime.now().isoformat()
                })
                socketio.emit('status', {'status': f'✨ Traditional transcription received ({result["language"]})'})

transcriber = EnhancedHybridTranscriber()

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
            max-width: 1200px;
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
        #start-traditional {
            background: linear-gradient(45deg, #2196F3, #1976D2);
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
        .transcription-output {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .output-panel {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            color: #333;
        }
        .transcription-area {
            width: 100%;
            height: 250px;
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
            .transcription-output {
                grid-template-columns: 1fr;
            }
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
            <button id="start-traditional">🎯 Start Traditional STT</button>
            <button id="stop" disabled>⏹️ Stop Recording</button>
            <button id="clear">🗑️ Clear All</button>
        </div>
        
        <div class="mode-panel" id="realtime-panel">
            <div class="status-text" id="realtime-status-text">Ready to connect...</div>
        </div>
        
        <div class="mode-panel" id="traditional-panel">
            <div class="status-text" id="traditional-status-text">Ready to start...</div>
        </div>
        
        <div class="transcription-output">
            <div class="output-panel">
                <div>Realtime Transcription</div>
                <textarea class="transcription-area" id="realtime-output" readonly
                    placeholder="Realtime transcriptions will appear here..."></textarea>
            </div>
            <div class="output-panel">
                <div>Traditional Transcription</div>
                <textarea class="transcription-area" id="traditional-output" readonly
                    placeholder="Traditional transcriptions will appear here..."></textarea>
            </div>
        </div>
        
        <div class="stats">
            <div>
                <div class="stat-value" id="realtime-words">0</div>
                <div>Realtime Words</div>
            </div>
            <div>
                <div class="stat-value" id="traditional-words">0</div>
                <div>Traditional Words</div>
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
        const socket = io();
        let mediaRecorder, stream, audioContext, processor;
        let isRecording = false;
        let sessionStartTime, currentMode;
        
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
                startTraditional: document.getElementById('start-traditional'),
                stop: document.getElementById('stop'),
                clear: document.getElementById('clear')
            };
            
            elements.startRealtime.addEventListener('click', () => startRecording('realtime'));
            elements.startTraditional.addEventListener('click', () => startRecording('traditional'));
            elements.stop.addEventListener('click', stopRecording);
            elements.clear.addEventListener('click', clearTranscriptions);
        });
        
        async function startRecording(mode) {
            if (isRecording) {
                debugLog('Already recording');
                return;
            }
            
            try {
                currentMode = mode;
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
                    socket.emit('audio_stream', {
                        audio: Array.from(samples),
                        mode: currentMode
                    });
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                sessionStartTime = Date.now();
                
                document.getElementById('start-realtime').disabled = true;
                document.getElementById('start-traditional').disabled = true;
                document.getElementById('stop').disabled = false;
                
                const panel = mode === 'realtime' ? 'realtime-panel' : 'traditional-panel';
                const statusText = mode === 'realtime' ? 'realtime-status-text' : 'traditional-status-text';
                document.getElementById(panel).classList.add('active');
                document.getElementById(statusText).textContent = mode === 'realtime'
                    ? 'Connecting to OpenAI...'
                    : 'Recording started...';
                    
                socket.emit(mode === 'realtime' ? 'start_realtime_streaming' : 'start_traditional_streaming');
                debugLog(`${mode} recording started`);
                
            } catch (err) {
                debugLog(`Error starting recording: ${err.message}`);
                alert(`Error: ${err.message}`);
            }
        }
        
        function stopRecording() {
            if (!isRecording) return;
            
            isRecording = false;
            sessionStartTime = null;
            
            if (processor) processor.disconnect();
            if (audioContext) audioContext.close();
            if (stream) stream.getTracks().forEach(track => track.stop());
            
            socket.emit('stop_streaming', { mode: currentMode });
            
            document.getElementById('start-realtime').disabled = false;
            document.getElementById('start-traditional').disabled = false;
            document.getElementById('stop').disabled = true;
            
            document.getElementById('realtime-panel').classList.remove('active');
            document.getElementById('traditional-panel').classList.remove('active');
            document.getElementById('realtime-status-text').textContent = 'Ready to connect...';
            document.getElementById('traditional-status-text').textContent = 'Ready to start...';
            
            debugLog('Recording stopped');
        }
        
        function clearTranscriptions() {
            document.getElementById('realtime-output').value = '';
            document.getElementById('traditional-output').value = '';
            document.getElementById('realtime-words').textContent = '0';
            document.getElementById('traditional-words').textContent = '0';
            debugLog('Transcriptions cleared');
        }
        
        socket.on('realtime_transcription', (data) => {
            const textarea = document.getElementById('realtime-output');
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            textarea.value += `[${timestamp}] [${data.language}] ${data.text}\n`;
            textarea.scrollTop = textarea.scrollHeight;
            
            const words = textarea.value.split(/\s+/).filter(word => word.length > 0);
            document.getElementById('realtime-words').textContent = words.length;
        });
        
        socket.on('transcription', (data) => {
            const textarea = document.getElementById('traditional-output');
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            textarea.value += `[${timestamp}] [${data.language}] ${data.text}\n`;
            textarea.scrollTop = textarea.scrollHeight;
            
            const words = textarea.value.split(/\s+/).filter(word => word.length > 0);
            document.getElementById('traditional-words').textContent = words.length;
        });
        
        socket.on('realtime_status', (data) => {
            document.getElementById('realtime-status-text').textContent = data.status || data.error;
            debugLog(`Realtime status: ${data.status || data.error}`);
        });
        
        socket.on('status', (data) => {
            document.getElementById('traditional-status-text').textContent = data.status;
            debugLog(`Traditional status: ${data.status}`);
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
    logger.info("Starting OpenAI Realtime streaming")
    try:
        transcriber.start_realtime()
        emit('realtime_status', {'status': '🔄 Connecting to OpenAI Realtime API...'})
    except Exception as e:
        logger.error(f"Error starting realtime streaming: {str(e)}")
        emit('realtime_status', {'error': f"❌ Failed to start: {str(e)}"})

@socketio.on('start_traditional_streaming')
def handle_start_traditional_streaming():
    logger.info("Starting traditional streaming")
    emit('status', {'status': '🎯 Traditional processing active'})

@socketio.on('audio_stream')
def handle_audio_stream(data):
    mode = data.get('mode', 'traditional')
    audio_samples = data.get('audio', [])
    
    try:
        transcriber.add_audio_chunk(audio_samples, mode)
    except Exception as e:
        logger.error(f"Error processing audio stream ({mode}): {str(e)}")

@socketio.on('stop_streaming')
def handle_stop_streaming(data):
    mode = data.get('mode', 'traditional')
    logger.info(f"Stopping {mode} streaming")
    
    try:
        transcriber.process_speech_end(mode)
        if mode == 'realtime':
            transcriber.stop_realtime()
            emit('realtime_status', {'status': '🔴 Realtime streaming stopped'})
        else:
            emit('status', {'status': '🔴 Traditional streaming stopped'})
    except Exception as e:
        logger.error(f"Error stopping streaming ({mode}): {str(e)}")

if __name__ == '__main__':
    try:
        logger.info(f"Using websockets version: {websockets.__version__}")
    except AttributeError:
        logger.warning("Could not determine websockets version")
    logger.info("Starting Bangla Speech-to-Text server...")
    socketio.run(
        app,
        host='0.0.0.0',
        port=7860,
        debug=False,
        allow_unsafe_werkzeug=True
    )