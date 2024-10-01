import os
import av
import argparse
import torch
import numpy as np
import random
from oneshot_clip_selector_gpt import GptSelectorOneShot
from FootageData import FootageData, Video
import math

class VideoMomentExtractorOneShot:
    def __init__(self, args):
        self.frame_extraction_frequency = args.frame_extraction_frequency
        self.args = args

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Initialize the one-shot selector model
        self.oneshot_selector = GptSelectorOneShot(os.getenv('OPENAI_API_KEY'), args.selectmodel, self.frame_extraction_frequency, args.user_prompt)
        self.target_duration_range = (30, 60)  # Range in seconds (min: 30, max: 60)
        self.seconds_per_clip = 4  # Assuming each clip is 4 seconds long

    def calculate_max_clips(self):
        """
        Calculate the maximum number of clips to select based on the target duration.
        """
        max_duration = self.target_duration_range[1]
        max_clips = math.floor(max_duration / self.seconds_per_clip)
        return max_clips

    def extract_frames_from_video(self, video_path, footage_data, max_frames=256):
        '''
        Extract frames and indices from a video at a given frame extraction frequency.
        Args:
            video_path (str): Path to the video file.
            footage_data (FootageData): Object to store video and frame data.
        '''
        print(f"Processing video: {video_path}")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        fps = container.streams.video[0].average_rate

        print(f"Total frames: {total_frames}")

        # Create a Video object
        video = Video(container=container, total_frames=total_frames, fps=fps, video_path=video_path)

        # Calculate indices to extract frames at a defined frequency
        indices = np.arange(0, total_frames, self.frame_extraction_frequency).astype(int)

        # Decode the video to extract frames
        container.seek(0)
        count = 0
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                image = frame.to_ndarray(format="rgb24")
                video.add_frame(image=image, frame_number=i)
                count += 1
            if count == max_frames:
                break

        # Add the processed video to the footage data
        footage_data.add_video(video)

        print(f"Extracted {len(video.frames)} frames from {video_path}.")

    def select_frames(self, footage_data):
        '''
        Compute the additional score for each frame based on 0.6 * visual score + 0.4 * relevance score,
        and select the top N clips based on the highest scores, ensuring no frame overlap.
        Args:
            footage_data (FootageData): Object storing video and frame data.
        Returns:
            selected_frames (list): List of selected frames.
        '''
        # Compute max number of clips based on target duration (e.g., 4 seconds per clip)
        max_clips = self.calculate_max_clips()

        # Compute the combined score for each frame
        scored_frames = []
        for video in footage_data.videos:
            for frame in video.frames:
                combined_score = 0.6 * frame.visual_score + 0.4 * frame.relatedness_score
                scored_frames.append((combined_score, frame))

        # Sort frames by combined score in descending order
        scored_frames.sort(reverse=True, key=lambda x: x[0])

        # Filter out 2 times the max number of selected clips from highest scoring clips
        top_candidates = scored_frames[:max_clips * 2]

        # Randomly select max number of clips, ensuring no overlap
        selected_frames = random.sample(top_candidates, max_clips)
        
        # Mark the selected frames as chosen
        for pair in selected_frames:
            pair[1].chosen = True

    def run(self, folder_path):
        '''
        Extract key moments and select the most interesting moments from all video files in a folder.
        Args:
            folder_path (str): Path to the folder containing video files.
        Returns:
            selected frames (list): List of selected frames.
        '''
        self.footage_data = FootageData(folder_path)

        # Get all video files in the folder
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print("No video files found in the folder.")
            return []

        # Process each video file
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            self.extract_frames_from_video(video_path, self.footage_data)

        # Use the one-shot selector to generate descriptions, scores, and select frames
        self.oneshot_selector.comprehend_and_select_frames(self.footage_data)

        # Select the top frames based on scores
        self.select_frames(self.footage_data)

# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using GptSelectorOneShot")
    parser.add_argument("--visionmodel", type=str, default="gpt-4", help="model for frame descriptions and selection")
    parser.add_argument("--selectmodel", type=str, default="gpt-4", help="model for selecting moments")
    parser.add_argument("--user_prompt", type=str, default="Create a scenic travel video that highlights natural landscapes.", help="User prompt for selecting moments.")
    parser.add_argument("--folder_path", type=str, default="testfootage", help="Path to the folder containing video files")
    parser.add_argument("--frame_extraction_frequency", type=int, default=48, help="Frequency of frame extraction (default: 48 frames)")
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()

    extractor = VideoMomentExtractorOneShot(args)
    selected_frames = extractor.run(args.folder_path)

    # For demonstration, print the selected frame details
    for frame in selected_frames:
        print(f"Frame ID: {frame.id}, Frame Number: {frame.frame_number}, Chosen: {frame.chosen}")
