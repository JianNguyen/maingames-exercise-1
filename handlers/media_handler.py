import gradio as gr
import torchaudio
import torchaudio.transforms as transforms
import torch
from handlers.speech2text import WhisperTranscriptor
from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector
from utils.youtube_handler import download_youtube_complete, get_youtube_id
import cv2
from PIL import Image
from io import BytesIO


class MediaHandler:
    def __init__(self):
        self.arc_model = WhisperTranscriptor()
        self.llm_model = GeminiLLM()
        self.pg_vector = PgVector()
        self.target_sample_rate = 16000
        self.frame_sample_rate = 50

    def process_local_media(self, video_path, kind_of="local", youtube_id=None):
        video_id = self.pg_vector.is_source_available(video_path)
        if video_id:
            return "Video has added, reload successfully", gr.update(visible=True), video_id
        try:
            waveform = self.extract_audio_to_array(video_path)
            images = self.extract_image_frames(video_path, self.frame_sample_rate)
            transcript, words_timestamps = self.arc_model.transcribe(waveform)

            summarized_context = self.llm_model.summarize(transcript)
            summarized_embedding = self.llm_model.get_embedding(summarized_context)
            embeds = self.llm_model.embed(transcript)  # split to chunk and embed

            # Insert to sources table
            duration = round(len(waveform) / self.target_sample_rate, 2)
            if kind_of == "local":
                video_id = self.pg_vector.insert_to_sources_tb(video_path, transcript, duration)
            elif kind_of == "youtube":
                if youtube_id is None:
                    raise "Need to specify youtube id"
                video_id = self.pg_vector.insert_to_sources_tb(youtube_id, transcript, duration)
            else:
                raise "Need to specify kind of source"

            # Insert to wordstimestamp table
            self.pg_vector.insert_words_timestamp_to_wordstimestamp_tb(video_id, words_timestamps)
            # Insert to embeddings table
            _ = self.pg_vector.insert_embedding_to_embeddings_tb(video_id, summarized_context, summarized_embedding)
            graph_nodes = self.pg_vector.insert_multiple_embeddings_to_embeddings_tb(video_id, embeds)
            # Insert to graph table
            self.pg_vector.create_graph_connections(graph_nodes)
            # Insert to imagesstorage table
            self.pg_vector.insert_image_to_imagesstorage_db(video_id=video_id, images=images)
        except Exception as e:
            delete_query = """
                    DELETE FROM sources WHERE id = %s;
                    """
            self.pg_vector.cursors.execute(delete_query, (video_id,))
            self.pg_vector.conn.commit()
            raise f"Error processing video: {str(e)}"


        return "Video processed successfully", gr.update(visible=True), video_id

    def process_youtube_media(self, youtube_link):
        youtube_id = get_youtube_id(youtube_link)
        video_id = self.pg_vector.is_source_available(youtube_id)
        if video_id:
            return "Video has added, reload successfully", gr.update(visible=True), video_id

        video_path = download_youtube_complete(youtube_link, output_path="downloads")
        return self.process_local_media(video_path, kind_of="youtube", youtube_id=youtube_id)

    def process(self, video_path, youtube_link):
        if video_path is not None:
            return self.process_local_media(video_path, kind_of="local")
        elif youtube_link is not None:
            return self.process_youtube_media(youtube_link)
        else:
            return "No media provided"


    def extract_audio_to_array(self, mp4_path):
        waveform, sample_rate = torchaudio.load(mp4_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if the sample rate is different

        if sample_rate != self.target_sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.numpy().flatten()

        return waveform

    def extract_image_frames(self, mp4_path, frame_sample_rate=50):
        cap = cv2.VideoCapture(mp4_path)
        frame_count = 0
        images_list = []
        while cap.isOpened():
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Finished reading video, extracted {len(images_list)} frames")
                        break

                    # Process every nth frame
                    if frame_count % frame_sample_rate == 0:
                        # Convert BGR (OpenCV) to RGB (PIL)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to bytes for database storage
                        pil_img = Image.fromarray(frame_rgb)
                        buffer = BytesIO()
                        pil_img.save(buffer, format="PNG")  # or JPEG
                        binary_image = buffer.getvalue()
                        images_list.append(binary_image)

                    frame_count += 1

            except Exception as e:
                raise(f"Error processing video: {str(e)}")

            finally:
                # Always release the video capture
                cap.release()

        return images_list