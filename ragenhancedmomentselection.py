import os
import av
import argparse
import random
import weaviate
from weaviate.classes.config import Configure
import io
import base64
import numpy as np
from FootageData import Frame
import requests
import string
from PIL import Image
import torch
import clip
from PIL import Image

class RAGEnhancedMomentSelector:
    def __init__(self, args):
        self.keyframe_frequency = args.keyframe_frequency
        self.args = args

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Connect to embedded Weaviate instance (local storage as backup)
        self.client = weaviate.connect_to_embedded(
            version="1.26.1",
            environment_variables={
                "ENABLE_MODULES": "backup-filesystem",
                "BACKUP_FILESYSTEM_PATH": "/home/alex/Documents/vscodeprojects/personal/aivideoedit/weaviate_backup_path",
            }
        )
        
        # Create collection to store frames
        self.client.collections.delete_all()
        self.client.collections.create(
            name="Frames",
            vectorizer_config=Configure.Vectorizer.none(),  # No external vectorizer
        )

        self.batch_size = 8
        self.keyframe_storage = []
        self.target_duration_range = (30, 60)  # Range in seconds (min: 30, max: 60)
        self.seconds_per_clip = 4  # Assuming each clip is 4 seconds long

        self.api_key = os.getenv('OPENAI_API_KEY')

    def terminate_session(self):
        self.client.close()

    def encode_image_clip(self, image):
        """
        Convert image into a vector using CLIP.
        """
        image_preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image_preprocessed)
        return image_embedding.cpu().numpy()

    def extract_frames_from_video(self, video_path):
        """
        Extract frames from video and store in Weaviate.
        """
        print(f"Processing video: {video_path}")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        fps = container.streams.video[0].average_rate

        container.seek(0)
        count = 0

        self.frames_collection = self.client.collections.get("Frames")
        with self.frames_collection.batch.dynamic() as batch:
            for i, frame in enumerate(container.decode(video=0)):
                print(f"Adding frame {i}")
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                
                # Use CLIP to embed the frame
                clip_embedding = self.encode_image_clip(image)

                frame_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                frame_data = {
                    "frameID": frame_id,
                    "frameNumber": i,
                    "filePath": video_path,
                }
                batch.add_object(properties=frame_data, vector=clip_embedding)

                count += 1
                
                if count % self.keyframe_frequency == 0:
                    self.keyframe_storage.append(image)

                if i == 20000:
                    break

        # Check for failed objects
        if len(self.frames_collection.batch.failed_objects) > 0:
            print(f"Failed to import {len(self.frames_collection.batch.failed_objects)} objects")
            for failed in self.frames_collection.batch.failed_objects:
                print(f"e.g. Failed to import object with error: {failed.message}")
        else:
            print("No errors")

    def encode_image_base64(self, image):
        """
        Convert a PIL image to a base64-encoded string.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save image as PNG to byte array
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def get_keyframe_descriptions(self):
        """
        Generate descriptions for each keyframe.
        Returns:
            list: A list of descriptions related to the video content.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        descriptions = []
        batch_size = 8

        # Process the keyframes in batches
        for batch_start in range(0, len(self.keyframe_storage), batch_size):
            batch_end = min(batch_start + batch_size, len(self.keyframe_storage))
            batch_frames = self.keyframe_storage[batch_start:batch_end]

            # Encode images into base64 strings
            base64_images = [self.encode_image_base64(image) for image in batch_frames]

            # Prepare the prompt for GPT
            content = [
                {
                    "type": "text",
                    "text": f"Describe each of the following {len(batch_frames)} images. Use less than 20 words per description."
                }
            ]
            content.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "low"}} for image in base64_images])

            # Prepare the request payload
            payload = {
                "model": self.args.gpt_model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 1024
            }

            try:
                # Send the request to GPT-4 API
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                if response.status_code == 200:
                    # Extract the descriptions from the response
                    batch_descriptions = response.json()['choices'][0]['message']['content']
                    descriptions.extend(batch_descriptions.strip().split('\n'))

                else:
                    print(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"Error describing frames {batch_start} to {batch_end - 1}: {e}")

        return descriptions

    def elaborate_on_prompt(self, descriptions):
        """
        Generate a list of interesting sights or actions that complement the user-provided prompt.
        Args:
            descriptions (list): The descriptions generated for the keyframes.
        Returns:
            list: A list of interesting sights or actions related to the video content.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Now generate a list of interesting sights or actions based on user prompt and descriptions
        description_prompt = f"Based on these keyframe descriptions below, and the user's prompt: '{self.args.user_prompt}', imagine some other sights or \
            actions that could've been part of the footage that fit the prompt and are visually interesting. in addition, choose from some of the \
            existing descriptions that are video worthy. make a list of these scenes and return each item on a new line\n  {descriptions}"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": description_prompt
                }
            ],
            "max_tokens": 512
        }

        try:
            # Send the request to GPT-4 API for generating the list
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                # Extract the generated list
                interesting_actions = response.json()['choices'][0]['message']['content']
                return interesting_actions.strip().split('\n')

            else:
                print(f"Error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            print(f"Error generating interesting actions: {e}")
            return []
        
    def generate_frame_description(self, frame_id, frame_image):
        """
        Generate a textual description for a frame using GPT.
        Args:
            frame_id (str): The frame ID.
            frame_image (PIL.Image): The image of the frame to describe.
        Returns:
            str: A textual description of the frame.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Encode the image to base64
        base64_image = self.encode_image_base64(frame_image)

        # Prepare the request payload for GPT
        content = [
            {
                "type": "text",
                "text": f"Describe this frame in detail."
            },
            {
                "type": "image_url",
                "image_url": { 
                    "url": f"data:image/png;base64,{base64_image}"                }
            }
        ]
        payload = {
                "model": self.args.gpt_model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 1024
            }

        try:
            # Send the request to GPT API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                # Extract the description from the response
                description = response.json()['choices'][0]['message']['content'].strip()
                return description
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return "Description generation failed."

        except Exception as e:
            print(f"Error generating description for frame {frame_id}: {e}")
            return "Description generation failed."
                        
    def query_vector_database(self, augmented_prompts):
        """
        For each augmented prompt, get the vector embedding using CLIP and query the Weaviate database for the best matching frame.
        For each chosen frame, generate a textual description using GPT, then return a list of Frame objects.
        Args:
            augmented_prompts (list): List of prompts generated from the keyframe descriptions.
        Returns:
            list: A list of Frame objects containing frame-specific information.
        """
        frames_list = []

        for prompt in augmented_prompts:
            print(f"Processing augmented prompt: {prompt}")
            
            # Get the CLIP embedding for the prompt
            text_tokens = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(text_tokens).cpu().numpy()

            # Weaviate vector search using the text embedding to get the best match
            search_result = self.frames_collection.query.near_vector(near_vector=text_embedding[0], limit=1)
            search_result = search_result.objects[0].properties

            # Open the video and retrieve the frame using PyAV
            container = av.open(search_result["filePath"])
            stream = container.streams.video[0]
            target_frame_number = int(search_result["frameNumber"])
            
            # Decode frames until the target frame is found
            container.seek(0)  # Start at the beginning of the video stream
            for i, frame in enumerate(container.decode(video=0)):
                if i == target_frame_number:
                    # Convert the frame to an image
                    frame_image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    # Generate a description for the frame
                    description = self.generate_frame_description(search_result["frameID"], frame_image)

                    # Create the Frame object
                    frame_obj = Frame(
                        id=search_result["frameID"],
                        image=frame_image,
                        frame_number=target_frame_number,
                        video_path=search_result["filePath"],
                        description=description,
                        chosen=True
                    )
                    
                    frames_list.append(frame_obj)
                    break

        return frames_list


    def run(self, folder_path):
        """
        Extract frames, store them in Weaviate, generate descriptions, and retrieve selected clips.
        """
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print("No video files found.")
            return

        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            self.extract_frames_from_video(video_path)
            
        # Get descriptions for the keyframes
        descriptions = self.get_keyframe_descriptions()
        print("\n\nkeyframe descriptions\n", descriptions)
        # Elaborate on the user's prompt using the descriptions
        augmented_prompts = self.elaborate_on_prompt(descriptions)
        print("\n\naugmented prompts\n", augmented_prompts)

        chosen_frames = self.query_vector_database(augmented_prompts)
        print("\n\nchosen frames")
        for frame in chosen_frames:
            print(f"Frame ID: {frame.id}, Frame Number: {frame.frame_number}, Description: {frame.description}")
        
        return chosen_frames
