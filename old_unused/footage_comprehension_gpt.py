import base64
import requests
from PIL import Image
import io

class GptFootageComprehension:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def encode_image_base64(self, image):
        """
        Convert a PIL image to a base64-encoded string.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Save image as PNG to byte array
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def comprehend_frames(self, footage_data):
        '''
        Process the frames using GPT-4 and set descriptions for each frame in the FootageData.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        '''
        print(f"Comprehending frames in {footage_data.num_videos()} videos using GPT-4...")

        batch_size = 8  # Define batch size
        # Iterate over each video in the footage data
        for video in footage_data.videos:
            print(f"Processing video with {len(video.frames)} frames.")

            # Process frames in batches of 4
            for batch_start in range(0, len(video.frames), batch_size):
                batch_end = min(batch_start + batch_size, len(video.frames))
                batch_frames = video.frames[batch_start:batch_end]

                print(f"Processing frames {batch_start} to {batch_end - 1}")

                # Convert each frame to base64-encoded images
                base64_images = [self.encode_image_base64(Image.fromarray(frame.image)) for frame in batch_frames]

                # Create the batch prompt, asking GPT-4 to describe all frames
                content = [
                    {
                        "type": "text",
                        "text": f"Describe each of the following {len(batch_frames)} images in terms of what is happening, stylistic attributes like color or lighting, and mood. use less than 20 words. Return each description on a new line."
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
                    "max_tokens": 1024
                }

                try:
                    # Send the request to GPT-4 API
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                    if response.status_code == 200:
                        # Extract the generated descriptions from the response
                        batch_descriptions = response.json()['choices'][0]['message']['content']
                        descriptions = []
                        for text in batch_descriptions.strip().split('\n'):
                            if len(text):
                                descriptions.append(text)
                        # print(descriptions)
                        # Assign descriptions to the corresponding frames
                        for i, frame in enumerate(batch_frames):
                            if i < len(descriptions):
                                frame.description = descriptions[i].strip()

                    else:
                        print(f"Error: {response.status_code} - {response.text}")

                except Exception as e:
                    print(f"Error describing frames {batch_start} to {batch_end - 1}: {e}")
