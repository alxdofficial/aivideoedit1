import os
import av
import torch
import numpy as np
import openai
from openai import OpenAI
import base64
import requests
import argparse
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import pprint
from PIL import Image
import io
import string
import random

class VideoMomentExtractor:
    def __init__(self, args):
        '''
        Initialize the VideoMomentExtractor.
        '''
        self.frame_extraction_frequency = args.frame_extraction_frequency
        self.args = args

        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # Use the first GPU
            print("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")  # Fall back to CPU
            print("CUDA is not available. Using CPU.")

        if args.visionmodel == 'llava':
            self.llavamodel = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            self.llavaprocessor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        else:
            self.gptmodel = args.visionmodel

        self.selectmodel = args.selectmodel

        # Retrieve API key from environment variables
        self.api_key = os.getenv('OPENAI_API_KEY')

        print("OpenAI API setup complete.")

    def extract_frames_from_video(self, video_path):
        '''
        Extract frames and indices from a video at a given frame extraction frequency.
        Args:
            video_path (str): Path to the video file.
        Returns:
            container (PyAV container): The PyAV container with video metadata.
            indices (list of int): The indices of the frames to be extracted.
            clip (np.ndarray): The extracted frames as an array.
            fps (float): The frame rate of the video.
            total_frames (int): The total number of frames in the video.
        '''
        print(f"Processing video: {video_path}")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        fps = container.streams.video[0].average_rate

        print(f"Total frames: {total_frames}, FPS: {fps}")

        # Calculate indices to extract frames at a defined frequency
        indices = np.arange(0, total_frames, self.frame_extraction_frequency).astype(int)

        # Decode the video to extract frames
        frames = []
        container.seek(0)
        print(f"Extracting frames at every {self.frame_extraction_frequency} frames...")
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame)

        clip = np.stack([x.to_ndarray(format="rgb24") for x in frames])
        print(f"Extracted {len(clip)} frames.")

        return container, indices, clip, fps, total_frames

    def encode_image_base64(self, image):
        """
        Convert a PIL image to a base64-encoded string.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save image as PNG to byte array
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def describe_frames_using_gpt(self, clip, indices, fps):
        '''
        Process the frames using GPT-4V via OpenAI API and extract descriptions for each frame.
        Args:
            clip (np.ndarray): Extracted frames as an array.
            indices (list of int): The indices of the frames.
            fps (float): Frame rate of the video.
        Returns:
            moments (list): A list of dicts with timestamps and frame descriptions.
        '''
        moments = []
        print(f"Describing {len(clip)} frames using {self.gptmodel}...")

        for idx, frame in zip(indices, clip):
            print("understanding frame: ", idx)
            # Convert the frame (as numpy array) to an image
            image = Image.fromarray(frame)

            # Convert the image to base64 string
            base64_image = self.encode_image_base64(image)

            # Prepare the request payload
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": f"{self.gptmodel}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "describe this image in terms of what is happening, \
                                stylistic attributes like color or lighting, and mood. your reponse should be concise and only include words relevant to the image. there is no need to be conversational"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }

            try:
                # Send the request to GPT-4V API
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                if response.status_code == 200:
                    # Extract the generated description from the response
                    description = response.json()['choices'][0]['message']['content']

                    # Store the result
                    moments.append({"id": ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)), "timestamp": idx, "description": description})
                else:
                    print(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                print(f"Error describing frame {idx}: {e}")

        return moments

    def describe_frames_using_llava(self, clip, indices, fps):
        '''
        Process the frames using LLaVA and extract descriptions for each frame.
        Args:
            clip (np.ndarray): Extracted frames as an array.
            indices (list of int): The indices of the frames.
            fps (float): Frame rate of the video.
        Returns:
            moments (list): A list of dicts with timestamps and frame descriptions.
        '''
        moments = []
        print(f"Describing {len(clip)} frames using LLaVA...")
        
        for idx, frame in zip(indices, clip):
            print("understanding frame: ", idx)
            # Create a single frame conversation prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is happening in this frame?"},
                        {"type": "image"}
                    ],
                }
            ]
            prompt = self.llavaprocessor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process the frame using the LLaVA model
            inputs = self.llavaprocessor(text=prompt, images=[frame], padding=True, return_tensors="pt").to(self.llavamodel.device)
            output = self.llavamodel.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            # Decode the output description
            description = self.llavaprocessor.decode(output[0][2:], skip_special_tokens=True)

            keyword = "ASSISTANT:"
            if keyword in description:
                description =  description.split(keyword, 1)[1].strip()
            
            
            # Store the result
            moments.append({"id": ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)), "frame": idx, "description": description})

        return moments

    def extract_video_moments(self, folder_path):
        '''
        Extracts key moments (frame descriptions with frame numbers) from all video files in a folder.
        Args:
            folder_path (str): Path to the folder containing video files.
        Returns:
            video_moments (dict): A dictionary with filenames as keys and moment descriptions with frame numbers, FPS, and total frames as values.
        '''
        video_moments = {}

        # Get all video files in the folder
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print("No video files found in the folder.")
            return video_moments

        # Process each video file
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            filename = video_file  # Extract filename from the path
            print("AI watching: ", filename)

            # Extract frames and indices
            container, indices, clip, fps, total_frames = self.extract_frames_from_video(video_path)

            # Describe the frames using GPT or LLaVA
            if self.args.visionmodel == "llava":
                moments = self.describe_frames_using_llava(clip, indices, fps)
            else:
                moments = self.describe_frames_using_gpt(clip, indices, fps)

            # Store the moments along with fps and total length
            video_moments[filename] = {
                "moments": moments,
                "fps": fps,
                "total_frames": total_frames
            }

        return video_moments
    
    def format_moments_for_prompt(self, video_moments_dict):
        '''
        Format the video moments dict into the prompt structure required by GPT-4O.
        Args:
            video_moments_dict (dict): The dictionary containing moments of videos with descriptions and timestamps.
        Returns:
            str: The formatted prompt string.
        '''
        formatted_clips = []
        for filename, data in video_moments_dict.items():
            for moment in data['moments']:
                clip_id = moment['id']
                description = moment['description']
                formatted_clips.append(f"- {clip_id}, {description}")

        formatted_prompt = (
            "available clips:\n" + "\n".join(formatted_clips) +
            "\n\ntask: please select a subset of the available clips based on the description that is the most "
            "interesting in terms of: fits together to form a narrative, visually pleasing, and minimizes duplication. "
            "return the results in the same format as available clips were given."
        )

        return formatted_prompt

    def select_moments(self, video_moments_dict):
        '''
        Format the moments into a prompt and ask GPT-4O to select the most interesting ones.
        Args:
            video_moments_dict (dict): The dictionary of video moments to be sent for selection.
        Returns:
            list: A list of selected clips in the same format as available clips.
        '''
        # Format the video moments into a prompt
        prompt = self.format_moments_for_prompt(video_moments_dict)

        # Prepare the request payload for GPT-4O
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": f"{self.selectmodel}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        }

        try:
            # Send the request to GPT-4O API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                # Extract the response and format it back into a list of selected moments
                selected_moments = response.json()['choices'][0]['message']['content']
                return selected_moments.strip().split("\n")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            print(f"Error in selecting moments: {e}")
            return []

# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using LLaVA or GPT-4V")
    # parser.add_argument("--visionmodel", type=str, default="gpt-4o-mini", help="model for frame descriptions")
    parser.add_argument("--visionmodel", type=str, default="llava", help="model for frame descriptions")
    parser.add_argument("--selectmodel", type=str, default="gpt-4o", help="model for selecting moments")

    parser.add_argument("--folder_path", type=str, default="testfootage0", help="Path to the folder containing video files")
    parser.add_argument("--frame_extraction_frequency", type=int, default=96, help="Frequency of frame extraction (default: 96 frames)")
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()

    extractor = VideoMomentExtractor(args)

    # Call the function to extract moments from the videos in the folder
    video_moments_dict = extractor.extract_video_moments(args.folder_path)
    print(video_moments_dict)

    # Select the most interesting moments using GPT-4O
    selected_moments = extractor.select_moments(video_moments_dict)
    print("Final selected moments:")
    print("\n".join(selected_moments))