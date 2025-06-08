from banglaspeech2text import Speech2Text
import gradio as gr

stt = Speech2Text()

gr.Interface(
    fn=stt.recognize,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text"
).launch(share=True)
