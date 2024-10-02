import requests
import math
import os

class Sequencer:
    def __init__(self, model_name):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

    def format_frames_for_gpt(self, footage_data):
        '''
        Format the selected frames into a structure for GPT-4O to order them.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        Returns:
            str: The formatted prompt string.
        '''
        selected_clips = []
        
        # Query all selected frames in all videos
        for video in footage_data.videos:
            for frame in video.frames:
                if frame.chosen:
                    selected_clips.append(f"- {frame.id}, {frame.description}")

        formatted_prompt = (
            f"You are helping to order clips for a video based on their visual interest and storytelling. "
            f"Please order the clips based on the descriptions to make a coherent, visually pleasing video.\n"
            "Return only the IDs of the clips, and nothing else, each id on a new line.\n"
            "\nAvailable clips:\n" + "\n".join(selected_clips)
        )

        return formatted_prompt

    def order_clips(self, footage_data):
        '''
        Queries GPT-4O to order selected frames based on visual interest and storytelling.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        Returns:
            list: List of ordered clip IDs.
        '''
        # Format the footage data into a prompt
        prompt = self.format_frames_for_gpt(footage_data)

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
                # Extract the response and parse the ordered clip IDs
                print(response.json()['choices'][0]['message']['content'])
                ordered_ids = response.json()['choices'][0]['message']['content'].strip().split("\n")
                return [clip_id.strip() for clip_id in ordered_ids]

            else:
                print(f"Error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            print(f"Error in ordering clips: {e}")
            return []

    def pretty_print_output(self, ordered_ids, footage_data):
        '''
        Pretty print the ordered clips with their ID, filepath, frame number, and timestamp.
        Args:
            ordered_ids (list): List of ordered clip IDs.
            footage_data (FootageData): The footage data containing videos and frames.
        '''
        print("\nFinal Ordered Clips:\n")
        for clip_id in ordered_ids:
            for video in footage_data.videos:
                for frame in video.frames:
                    if frame.id == clip_id:
                        # Calculate timestamp from frame number and frame rate
                        timestamp = frame.frame_number / video.fps
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        timestamp_str = f"{minutes:02}:{seconds:02}"

                        print(f"Clip ID: {frame.id}")
                        print(f"Filepath: {video.video_path}")
                        print(f"Frame Number: {frame.frame_number}")
                        print(f"Timestamp: {timestamp_str}")
                        print("-" * 40)

    def run(self, footage_data):
        '''
        Runs the complete process of ordering clips and printing the final ordered list.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        '''
        # Step 1: Order the clips using GPT-4O based on visual interest and storytelling
        ordered_ids = self.order_clips(footage_data)

        # Step 2: Pretty print the ordered clips with relevant details
        if ordered_ids:
            self.pretty_print_output(ordered_ids, footage_data)
        else:
            print("No clips were ordered.")
