import gradio as gr
from markdown_it.cli.parse import interactive


def chatbot_response(message, history, image=None):
    # Process the inputs and generate a response
    # This is a placeholder for your actual processing logic
    return f"You asked about: {message}. I've analyzed the provided media."


def toggle_image_visibility(current_state):
    # Toggle the visibility: if currently visible, hide it; if hidden, show it
    new_state = not current_state  # Flip the current state
    return gr.update(visible=new_state, interactive=new_state), new_state  # Update




with gr.Blocks(css="""
    .gradio-container {
        margin: auto;
        width: 100%;
    }
    .chatbot-container {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .chat-message-user {
        background-color: #f7f7f8;
    }
    .chat-message-bot {
        background-color: #ffffff;
    }
""") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Media Inputs")
            video_input = gr.File(label="Upload MP4 Video", type="filepath")
            gr.Markdown("### OR")
            youtube_input = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=example")
            # image_input = gr.Image(label="Upload Image (Optional)", type="filepath")

            # Add a button to process media before chatting
            process_button = gr.Button("Process Media", variant="secondary")
            status_text = gr.Textbox(label="Status", value="Ready to process media", interactive=False)

        with gr.Column(scale=3, elem_classes="chatbot-container"):
            gr.Markdown("# 🎥 Video Analysis Chat")
            chat = gr.ChatInterface(
                fn=chatbot_response,
                additional_inputs=[],
                examples=[
                    ["What is the main topic of this video?"],
                    ["Can you summarize the key points?"],
                    ["Who are the main speakers in this video?"]
                ],
                title="",
                description="Chat with your media files. Ask questions about the content.",
                type="messages",

            )
            with gr.Column():
                gr.Markdown("## Click to Show Image")
                # Button to trigger the image display
                show_image_button = gr.Button("Show Image")
                # Hidden Image element, initially empty
                # additional_image = gr.Image(label="Upload Image", sources=["upload"], type="filepath", visible=False)
                image_visibility_state = gr.State(False)
                additional_image = gr.Image(label="Upload Image", type="filepath", visible=False)

            chat.textbox.visible = False
    show_image_button.click(
        toggle_image_visibility,  # Function to call
        inputs=[image_visibility_state]
,  # No inputs are required
        outputs=[additional_image, image_visibility_state]
  # Update the visibility of the image
    )


    # You can add function to process the media when the process button is clicked
    def process_media(video, youtube_link, image):
        # Here you would process the media files before chatting
        # This is a placeholder for your actual processing logic

        if video is not None:
            return "Video processed successfully"
        elif youtube_link:
            return f"YouTube video from {youtube_link} processed"
        elif image is not None:
            return "Image processed successfully"
        else:
            return "No media provided"


    process_button.click(
        process_media,
        inputs=[video_input, youtube_input],
        outputs=[status_text]
    )

demo.launch(debug=True)
