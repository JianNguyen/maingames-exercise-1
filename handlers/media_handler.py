import gradio as gr
import torchaudio
import torchaudio.transforms as transforms
import torch
from handlers.speech2text import WhisperTranscriptor
from services.gemini_llm import GeminiLLM
from services.pgvector.connector import PgVector


class MediaHandler:
    def __init__(self):
        self.arc_model = WhisperTranscriptor()
        self.llm_model = GeminiLLM()
        self.pg_vector = PgVector()
        self.target_sample_rate = 16000

    def process_media(self, video_path, youtube_link):
        print(video_path, youtube_link)
        if video_path is not None:
            video_id = self.pg_vector.is_source_available(video_path)
            if video_id:
                return "Video has added, reload successfully", gr.update(visible=True), video_id

            waveform = self.extract_audio_to_array(video_path)
            transcript, words_timestamps = self.arc_model.transcribe(waveform)

            summarized_context = self.llm_model.summarize(transcript)
            summarized_embedding = self.llm_model.get_embedding(summarized_context)
            embeds = self.llm_model.embed(transcript) # split to chunk and embed
            # Insert to sources table
            video_id = self.pg_vector.insert_to_sources_tb(video_path, transcript)
            # Insert to wordstimestamp table
            self.pg_vector.insert_words_timestamp_to_wordstimestamp_tb(video_id, words_timestamps)
            # Insert to embeddings table
            _ = self.pg_vector.insert_embedding_to_embeddings_tb(video_id, summarized_context, summarized_embedding)
            graph_nodes = self.pg_vector.insert_multiple_embeddings_to_embeddings_tb(video_id, embeds)
            # Insert to graph table
            self.pg_vector.create_graph_connections(graph_nodes)
            return "Video processed successfully", gr.update(visible=True), video_id
        elif youtube_link is not None:
            return f"YouTube video from {youtube_link} processed"
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


    # sf.write(audio_buffer, audio_array, samplerate=44100, format="wav")
    # audio_buffer.seek(0)  # Reset buffer position
    #
    # # ---- Extract Frames ----
    # frames = []
    # cap = cv2.VideoCapture(mp4_path)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frames.append(frame)  # Store frame in memory
    #
    # cap.release()