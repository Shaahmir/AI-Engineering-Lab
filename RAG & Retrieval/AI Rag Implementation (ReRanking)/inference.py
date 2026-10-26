import gradio as gr
from implementation.answer import call_llm

interface = gr.ChatInterface(
    call_llm
)

interface.launch(
    inbrowser = True
)