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

        # Iterate over each video in the footage data
        for video in footage_data.videos:
            print(f"Processing video with {len(video.frames)} frames.")
            for frame in video.frames:
                print(f"Processing frame {frame.frame_number}")

                # Convert the frame (as numpy array) to an image
                image = Image.fromarray(frame.image)

                # Convert the image to base64 string
                base64_image = self.encode_image_base64(image)

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
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in terms of what is happening, stylistic attributes like color or lighting, and mood."
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
                    # Send the request to GPT-4 API
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                    if response.status_code == 200:
                        # Extract the generated description from the response
                        description = response.json()['choices'][0]['message']['content']

                        # Set the description field for the current frame
                        frame.description = description
                    else:
                        print(f"Error: {response.status_code} - {response.text}")

                except Exception as e:
                    print(f"Error describing frame {frame.frame_number}: {e}")
