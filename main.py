import os
import re
import time

import gradio as gr
from agents.multiple_agents import MultipleAgents
from handlers.media_handler import MediaHandler
from langchain_core.messages import HumanMessage
from utils.youtube_handler import get_youtube_title

from dotenv import load_dotenv
load_dotenv()

_agents_cache = {}
def init_multiple_agents(_id):
    print(f"Init_multiple_agents: {_id}")
    multiple_agents = MultipleAgents(_id)
    multiple_agents.build()
    _agents_cache[_id] = multiple_agents

def validate_youtube_url(url):
    if not url:
        return ""
    youtube_pattern = re.compile(r'^(https?:\/\/)?(www\.)?youtube\.com\/.*')
    if youtube_pattern.match(url):
        return url
    else:
        return ""

def get_title_video(path, kind_of):
    if kind_of == "video":
        title = os.path.basename(path)
    elif kind_of == "youtube":
        title = get_youtube_title(path)
    else:
        raise ValueError("Invalid kind of file")
    return title


def chatbot_response(message, history, _id, image=None):
    image_path = image
    initial_state = {"messages": [HumanMessage(content=message, additional_kwargs={"history": history,
                                                                                   "image": image_path})]}
    response = _agents_cache[_id].graph.invoke(initial_state)
    return response["messages"][-1].content, None


def process_media(video_input, youtube_input):
    if video_input:
        title = get_title_video(video_input, "video")
    elif youtube_input:
        title = get_title_video(youtube_input, "youtube")
    else:
        return "No input provided yet", "", "", []

    status, chat_text, vid_id = MediaHandler().process(video_input, youtube_input)
    messages = [{'role': 'assistant',
                 'metadata': None,
                 'content': f'Hello !! Welcome to Eklipse BOT\nIf you have any questions regarding <strong>{title}</strong>, feel free to ask !',
                 'options': None}]
    time.sleep(3) # rest time for init lang graph
    return status, chat_text, vid_id, messages


def toggle_image_visibility(current_state):
    new_state = not current_state
    return gr.update(visible=new_state, interactive=new_state), new_state  # Update


if __name__ == "__main__":

    additional_image = gr.Image(label="Upload Image", type="filepath", visible=True)

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
                video_input = gr.File(label="Upload MP4 Video", type="filepath", file_types=[".mp4"])
                gr.Markdown("### OR")
                youtube_input = gr.Textbox(
                    label="YouTube Link",
                    placeholder="https://www.youtube.com/watch?v=example",
                    info="Please enter a valid YouTube URL (youtube.com only)"
                )
                youtube_input.change(
                    fn=validate_youtube_url,
                    inputs=youtube_input,
                    outputs=[youtube_input]
                )

                process_button = gr.Button("Process Media", variant="secondary")
                status_text = gr.Textbox(label="Status", value="Ready to process media", interactive=False)
                video_id = gr.State("")


                def lock_youtube_when_video_exists(video_path):
                    """Disable YouTube input when video is uploaded"""
                    if video_path is not None:
                        # If video is uploaded, disable YouTube input
                        return gr.update(interactive=False, value="",
                                         placeholder="Video file uploaded - clear video to use YouTube")
                    else:
                        # If no video, enable YouTube input
                        return gr.update(interactive=True, placeholder="https://www.youtube.com/watch?v=example")


                def lock_video_when_youtube_exists(youtube_url):
                    """Disable video upload when YouTube URL is entered"""
                    if youtube_url and youtube_url.strip():
                        # If YouTube URL exists, disable video upload
                        return gr.update(interactive=False, value=None)
                    else:
                        # If no YouTube URL, enable video upload
                        return gr.update(interactive=True)


                def update_status(video_path, youtube_url):
                    if video_path is not None:
                        return "Video file ready for processing"
                    elif youtube_url and youtube_url.strip():
                        return "YouTube URL ready for processing"
                    else:
                        return "No input provided yet"


                video_input.change(
                    fn=lock_youtube_when_video_exists,
                    inputs=video_input,
                    outputs=youtube_input
                )

                youtube_input.change(
                    fn=lock_video_when_youtube_exists,
                    inputs=youtube_input,
                    outputs=video_input
                )

                video_input.change(
                    fn=update_status,
                    inputs=[video_input, youtube_input],
                    outputs=status_text
                )

                youtube_input.change(
                    fn=update_status,
                    inputs=[video_input, youtube_input],
                    outputs=status_text
                )
                video_id.change(
                    fn=init_multiple_agents,
                    inputs=[video_id],
                    outputs=[]
                )

            with gr.Column(scale=3, elem_classes="chatbot-container"):
                gr.Markdown("# 🎥 Video Analysis Chat")
                chat = gr.ChatInterface(
                    fn=chatbot_response,
                    additional_inputs=[video_id, additional_image],
                    additional_outputs=[additional_image],
                    title="",
                    description="Chat with your media files. Ask questions about the content.",
                    type="messages",
                    chatbot=gr.Chatbot(placeholder="<strong>Your Personal</strong><br>Ask Me Anything")

                )
                chat.textbox.visible = False

        process_button.click(fn=lambda: gr.update(interactive=False),
                            outputs=process_button
                            ).then(
                            process_media,
                            inputs=[video_input, youtube_input],
                            outputs=[status_text, chat.textbox, video_id, chat.chatbot],
        )

    demo.launch(debug=True)
