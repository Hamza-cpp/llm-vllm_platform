import gradio as gr
import requests

def generate_response(context, user_question, model):
    url = "http://localhost:8000/api/generate"
    data = {"context": context, "user_question": user_question, "model": model}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['response'], ""
    return "Failed to connect to the server", ""

def submit_rating(response_text, rating):
    if rating is not None and response_text != "":
        url = "http://localhost:8000/api/save_rating"
        data = {"response": response_text, "rating": rating}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return "Rating submitted successfully!"
        return f"Failed to submit rating: {response.status_code}"
    return "No rating submitted"

with gr.Blocks() as iface:
    with gr.Row():
        context_input = gr.Textbox(label="Context")
        question_input = gr.Textbox(label="User Question")
        model_input = gr.Dropdown(label="Model", choices=["phi3.5:3.8b", "qwen2.5:3b", "qwen2.5:1.5b", "phi:2.7b","qwen2.5:0.5b", "tinyllama:1.1b"])
        submit_btn = gr.Button("Generate Response")

    response_output = gr.Textbox(label="Response", interactive=False)
    rating_input = gr.Slider(label="Rate the response", minimum=1, maximum=5, step=1)
    rating_btn = gr.Button("Submit Rating")
    rating_output = gr.Label()

    submit_btn.click(
        generate_response,
        inputs=[context_input, question_input, model_input],
        outputs=[response_output, rating_input]
    )

    rating_btn.click(
        submit_rating,
        inputs=[response_output, rating_input],
        outputs=rating_output
    )

iface.launch()