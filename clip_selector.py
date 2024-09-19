import requests
import math

class ClipSelector:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        # self.user_prompt = "Create a scenic travel video that highlights natural landscapes, adventure, and moments of serenity."
        self.user_prompt = "Create a energetic city lifestyle travel video showing friends and fun moments"
        self.target_duration_range = (30, 60)  # Range in seconds (min: 30, max: 60)
        self.seconds_per_clip = 4  # Assuming each clip is 4 seconds long

    def calculate_max_clips(self):
        """
        Calculate the maximum number of clips to select based on the target duration.
        """
        max_duration = self.target_duration_range[1]
        max_clips = math.floor(max_duration / self.seconds_per_clip)
        return max_clips

    def format_moments_for_prompt(self, footage_data, max_clips):
        '''
        Format the FootageData into the prompt structure required by GPT-4O, including user prompt and clip limit.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
            max_clips (int): Maximum number of clips to select.
        Returns:
            str: The formatted prompt string.
        '''
        formatted_clips = []
        for video in footage_data.videos:
            for frame in video.frames:
                clip_id = frame.id
                description = frame.description
                formatted_clips.append(f"- {clip_id}, {description}")

        formatted_prompt = (
            f"You are helping select clips for a video based on the following prompt: '{self.user_prompt}'.\n"
            f"The video should be between 30 seconds to 1 minute long, and each clip is approximately 4 seconds.\n"
            f"Select up to {max_clips} clips, prioritizing those that fit together to form a narrative, are visually "
            f"pleasing, and minimize duplication. Feel free to select fewer clips to avoid redundancy.\n"
            "\nReturn the selected clip ids in the following comma-separated and no-space format: A1B2C3,A1B2C3,A1B2C3..."
            f"\nAvailable clips:\n" + "\n".join(formatted_clips) 
            )

        return formatted_prompt

    def select_moments(self, footage_data):
        '''
        Format the moments into a prompt and ask GPT-4O to select the most interesting ones.
        Args:
            footage_data (FootageData): The footage data to be sent for selection.
        Returns:
            None: Updates the `chosen` boolean field for each frame in the footage data.
        '''
        # Calculate the maximum number of clips to select
        max_clips = self.calculate_max_clips()

        # Format the footage data into a prompt
        prompt = self.format_moments_for_prompt(footage_data, max_clips)

        # Prepare the request payload for GPT-4O
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
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 2048
        }

        try:
            # Send the request to GPT-4O API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                # Extract the response and convert selected moment ids into a set for faster lookup
                selected_moment_ids = set()
                for moment_id in response.json()['choices'][0]['message']['content'].split(","):
                    selected_moment_ids.add(moment_id.strip())
                print("Chosen IDs: ", selected_moment_ids)

                # Iterate through the footage data and set the chosen boolean to True if the frame id is in the set
                for video in footage_data.videos:
                    for frame in video.frames:
                        frame.chosen = frame.id in selected_moment_ids

            else:
                print(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error in selecting moments: {e}")
