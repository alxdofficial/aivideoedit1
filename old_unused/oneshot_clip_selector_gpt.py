import base64
import requests
from PIL import Image
import io
import math
import json

class GptSelectorOneShot:
    def __init__(self, api_key, model_name, user_prompt):
        self.api_key = api_key
        self.model_name = model_name
        self.user_prompt = user_prompt

    def encode_image_base64(self, image):
        """
        Convert a PIL image to a base64-encoded string.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save image as PNG to byte array
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def comprehend_and_select_frames(self, footage_data):
        '''
        Process the frames using GPT-4, generate descriptions, and assign scores for prompt-relatedness and visual interest.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        '''
        print(f"Processing and selecting frames in {footage_data.num_videos()} videos using GPT-4...")

        batch_size = 8  # Define batch size

        # Iterate over each video in the footage data
        for video in footage_data.videos:
            print(f"Processing video with {len(video.frames)} frames.")

            # Process frames in batches of 8
            for batch_start in range(0, len(video.frames), batch_size):
                batch_end = min(batch_start + batch_size, len(video.frames))
                batch_frames = video.frames[batch_start:batch_end]

                print(f"Processing frames {batch_start} to {batch_end - 1}")

                # Convert each frame to base64-encoded images
                base64_images = [self.encode_image_base64(Image.fromarray(frame.image)) for frame in batch_frames]

                # Create the batch prompt, asking GPT-4 to describe all frames and score them
                content = [
                    {
                        "type": "text",
                        "text": (
                            f"Describe each of the following {len(batch_frames)} images based on the user prompt: '{self.user_prompt}'. "
                            f"For each image, return a JSON object in the format: "
                            f'{{"description": "<text>", "relatedness_score": <0-100>, "visual_score": <0-100>}}. '
                            f"Make sure each JSON is on a new line for each image. do not include any formatting hints in your response"
                        )
                    }
                ]
                content.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "low"}} for image in base64_images])

                # Prepare the request payload
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                payload = {
                    "model": f"{self.model_name}",
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "max_tokens": 2048
                }

                try:
                    # Send the request to GPT-4 API
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                    if response.status_code == 200:
                        # Extract the generated descriptions and scores from the response
                        batch_responses = response.json()['choices'][0]['message']['content']
                        print(batch_responses)
                        # Split the batch response into lines
                        parsed_responses = batch_responses.strip().split('\n')

                        # Assign scores and descriptions to the corresponding frames
                        for i, line in enumerate(parsed_responses):
                            if i < len(batch_frames):
                                try:
                                    # Parse the JSON object for each frame
                                    json_response = json.loads(line.strip())
                                    batch_frames[i].description = json_response["description"].strip()
                                    batch_frames[i].relatedness_score = int(json_response["relatedness_score"])
                                    batch_frames[i].visual_score = int(json_response["visual_score"])
                                except:
                                    print("line failed to parse: ", line)
                                    pass

                    else:
                        print(f"Error: {response.status_code} - {response.text}")

                except Exception as e:
                    print(f"Error describing frames {batch_start} to {batch_end - 1}: {e}")
