import requests
import math
import re

class ClipSelector:
    def __init__(self, api_key, model_name, frame_extraction_frequency, user_prompt):
        self.api_key = api_key
        self.model_name = model_name
        self.user_prompt = user_prompt

        self.target_duration_range = (30, 60)  # Range in seconds (min: 30, max: 60)
        self.seconds_per_clip = 4  # Assuming each clip is 4 seconds long
        self.frame_extraction_frequency = frame_extraction_frequency  # Sampling frame rate (e.g., 48)

    def calculate_max_clips(self):
        """
        Calculate the maximum number of clips to select based on the target duration.
        """
        max_duration = self.target_duration_range[1]
        max_clips = math.floor(max_duration / self.seconds_per_clip)
        return max_clips

    def calculate_min_interval(self):
        """
        Calculate the minimum interval between selected frames (in terms of frame IDs) to ensure that
        each selected keyframe is surrounded by unselected frames.
        """

        # Since the selected frame is the center, the buffer on either side is half of frames_per_clip, assume clips have mean framerate of 24.
        return self.seconds_per_clip * 24 // self.frame_extraction_frequency // 2 

    def format_moments_for_prompt(self, footage_data, max_clips, min_interval):
        '''
        Format the FootageData into the prompt structure required by GPT-4O, including user prompt and clip limit.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
            max_clips (int): Maximum number of clips to select.
            min_interval (int): Minimum interval between selected frame IDs.
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
            f"Select up to {max_clips} clips, prioritizing those that fit together to form a narrative, are visually "
            f"pleasing, and minimize duplication. Ensure there is at least {min_interval} frame IDs between selected clips "
            f"to avoid selecting frames that are too close together.\n"
            "\nReturn the selected clip ids on a new line"
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

        # Calculate the minimum interval between selected frame IDs
        min_interval = self.calculate_min_interval()
        # print("min interval: ", min_interval)

        # Format the footage data into a prompt
        prompt = self.format_moments_for_prompt(footage_data, max_clips, min_interval)

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
            "max_tokens": 1024
        }

        try:
            # Send the request to GPT-4O API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                # Extract the response and convert selected moment ids into a set for faster lookup
                selected_moment_ids = set()
                valid_id_pattern = re.compile(r"[A-Z0-9]{6}")  # Regex for validating IDs with 6 characters (A-Z, 0-9)
                print(response.json()['choices'][0]['message']['content'])
                # Split response content and add only valid IDs to the set
                for response in response.json()['choices'][0]['message']['content'].split("\n"):
                    response = response.strip()
                    # print(response)
                    formatted_ids = valid_id_pattern.findall(response)
                    # print("formatted: ", formatted_ids)
                    # Add only valid matches to the set
                    for formatted_id in formatted_ids:
                        selected_moment_ids.add(formatted_id)

                print("Chosen valid IDs: ", selected_moment_ids)

                # Iterate through the footage data and set the chosen boolean to True if the frame id is in the set
                for video in footage_data.videos:
                    for frame_idx, frame in enumerate(video.frames):
                        if frame.id in selected_moment_ids:
                            frame.chosen = True
                            # if frame in min interval already chosen, dont choose this one
                            for i in range(max(frame_idx - min_interval,0), frame_idx):
                                if video.frames[i].chosen:
                                    frame.chosen = False
                            for i in range(frame_idx + 1, min(len(video.frames), frame_idx + min_interval)):
                                if video.frames[i].chosen:
                                    frame.chosen = False
                        else:
                            frame.chosen = False

            else:
                print(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error in selecting moments: {e}")

