from banglaspeech2text import Speech2Text
from banglaspeech2text.cli import use_mic

models = Speech2Text.list_models()
print(models)

stt = Speech2Text("tiny")  # tiny, base, small

# Use with file
path = "test/test.wav"
text = stt.recognize(path)
print("Transcription:", text)
# Get segments
segments = stt.recognize(path, return_segments=True)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Use with microphone programatically (only for testing)
use_mic(stt)
