from io import BytesIO
from services.pgvector.connector import PgVector
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import math


class ImageHandler:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.pg_vector = PgVector()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_image_feature(self, image:"str|Image", text_prompt):
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt]
        # Extract features from the query image with the text prompt
        if isinstance(image, str):
            image = Image.open(image)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_outputs = self.model(**inputs)

        # Store query image embeddings
        query_image_features = query_outputs.image_embeds
        return query_image_features

    def search_image(self, video_id, query_image, text_prompt, similarity_threshold=0.7,):
        matches = []
        self.pg_vector.cursors.execute("SELECT duration FROM sources WHERE id = %s", (video_id,))
        duration = self.pg_vector.cursors.fetchall()[0][0]
        self.pg_vector.cursors.execute("SELECT id, image_data FROM imagesstorage WHERE video_id = %s", (video_id,))
        results = self.pg_vector.cursors.fetchall()

        query_image_features = self.get_image_feature(image=query_image, text_prompt=text_prompt)
        self.scale = duration / len(results)
        for result in results:
            _id, image_data = result
            frame_pil = Image.open(BytesIO(image_data))
            # Get frame image features
            frame_image_features = self.get_image_feature(image=frame_pil, text_prompt=text_prompt)
            # Calculate similarity between query and frame
            similarity = torch.nn.functional.cosine_similarity(
                query_image_features, frame_image_features
            ).item()
            # Check if similarity exceeds threshold
            if similarity > similarity_threshold:
                matches.append(_id)
        return matches

    def seconds_to_hms(self, seconds):
        seconds = math.floor(seconds)  # Truncate fractional seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        elif minutes > 0:
            return f"{minutes}:{seconds:02d}"
        else:
            return f"{seconds} seconds"

    def handle_matches(self, matches):
        results = []
        start_id = matches[0]
        for i in range(len(matches)):
            if i == len(matches) - 1:
                end_id = matches[i]
                results.append((start_id, end_id))
                break
            if matches[i] + 1 != matches[i + 1]:
                end_id = matches[i]
                results.append((start_id, end_id))
                start_id = matches[i + 1]
        text_response = "The time frames that appear:\n"
        for i, result in enumerate(results, 1):
            sec_start = self.seconds_to_hms(result[0]*self.scale)
            sec_end = self.seconds_to_hms(result[1]*self.scale)
            text_response += f"Scene {i}: Start: {sec_start} - End: {sec_end}\n"
        return text_response