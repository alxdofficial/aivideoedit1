import requests
import os

class Sequencer:
    def __init__(self, model_name):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

    def format_frames_for_gpt(self, frames):
        '''
        Format the selected frames into a structure for GPT-4O to order them.
        Args:
            frames (list): List of selected Frame objects.
        Returns:
            str: The formatted prompt string.
        '''
        selected_clips = []
        
        # Query all selected frames
        for frame in frames:
            if frame.chosen:
                selected_clips.append(f"- {frame.id}, {frame.description}")

        formatted_prompt = (
            f"You are helping to order clips for a video based on their visual interest and storytelling. "
            f"Please order the clips based on the descriptions to make a coherent, visually pleasing video.\n"
            "Return only the IDs of the clips, and nothing else, each id on a new line.\n"
            "\nAvailable clips:\n" + "\n".join(selected_clips)
        )

        return formatted_prompt

    def order_clips(self, frames):
        '''
        Queries GPT-4O to order selected frames based on visual interest and storytelling.
        Args:
            frames (list): List of selected Frame objects.
        Returns:
            list: List of ordered clip IDs.
        '''
        # Format the frames into a prompt
        prompt = self.format_frames_for_gpt(frames)

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

    def pretty_print_output(self, ordered_ids, frames):
        '''
        Pretty print the ordered clips with their ID, filepath, and frame number.
        Args:
            ordered_ids (list): List of ordered clip IDs.
            frames (list): List of selected Frame objects.
        '''
        print("\nFinal Ordered Clips:\n")
        for clip_id in ordered_ids:
            for frame in frames:
                if frame.id == clip_id:
                    print(f"Clip ID: {frame.id}")
                    print(f"Filepath: {frame.video_path}")
                    print(f"Frame Number: {frame.frame_number}")
                    print(f"Description: {frame.description}")
                    print("-" * 40)

    def run(self, frames):
        '''
        Runs the complete process of ordering clips and printing the final ordered list.
        Args:
            frames (list): List of selected Frame objects.
        '''
        print(frames)
        # Step 1: Order the clips using GPT-4O based on visual interest and storytelling
        ordered_ids = self.order_clips(frames)

        # Step 2: Pretty print the ordered clips with relevant details
        if ordered_ids:
            self.pretty_print_output(ordered_ids, frames)
        else:
            print("No clips were ordered.")

        return ordered_ids

       