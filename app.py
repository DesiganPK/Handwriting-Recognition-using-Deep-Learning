import gradio as gr 
from predict import input  
from inferenceModel4 import output


def combine(image):
    print(image)
    input(image)
    textReturn, text = output()
    return textReturn, text


with gr.Blocks() as demo: 
    input1 = gr.Image(label="Input Image", type="filepath")
    output1 = gr.Textbox(label="Predicted text")
    output2 = gr.Textbox(label="GenAI Text")

    submitBt = gr.Button("Submit")
    submitBt.click(fn=combine, inputs=input1, outputs=[output1, output2])


demo.launch(share=True)