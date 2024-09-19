import os
import av
import argparse
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from footage_comprehension_llava import LlavaFootageComprehension
from footage_comprehension_gpt import GptFootageComprehension
from clip_selector import ClipSelector
from FootageData import FootageData, Video
from visualizeselection import FootageVisualizer


class VideoMomentExtractor:
    def __init__(self, args):
        self.frame_extraction_frequency = args.frame_extraction_frequency
        self.args = args

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Initialize models based on args
        if args.visionmodel == 'llava':
            llava_model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            llava_processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
            self.footage_comprehension = LlavaFootageComprehension(llava_model, llava_processor, self.device)
        else:
            self.footage_comprehension = GptFootageComprehension(api_key=os.getenv('OPENAI_API_KEY'), model_name=args.visionmodel)

        self.clip_selector = ClipSelector(api_key=os.getenv('OPENAI_API_KEY'), model_name=args.selectmodel)

    def extract_frames_from_video(self, video_path, footage_data):
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
        video = Video(container=container, total_frames=total_frames, fps=fps)

        # Calculate indices to extract frames at a defined frequency
        indices = np.arange(0, total_frames, self.frame_extraction_frequency).astype(int)

        # Decode the video to extract frames
        container.seek(0)
        # print(f"Extracting frames at every {self.frame_extraction_frequency} frames...")
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                image = frame.to_ndarray(format="rgb24")
                video.add_frame(image=image, frame_number=i)

        # Add the processed video to the footage data
        footage_data.add_video(video)

        print(f"Extracted {len(video.frames)} frames from {video_path}.")

    def run(self, folder_path):
        '''
        Extracts key moments (frame descriptions with frame numbers) from all video files in a folder.
        Args:
            folder_path (str): Path to the folder containing video files.
        Returns:
            selected moments
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

        # use vlm to textually describe moments
        self.footage_comprehension.comprehend_frames(self.footage_data)

        # Select the most interesting moments using GPT-4O
        final_selected = self.clip_selector.select_moments(self.footage_data)


# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using LLaVA or GPT-4V")
    parser.add_argument("--visionmodel", type=str, default="llava", help="model for frame descriptions")
    # parser.add_argument("--visionmodel", type=str, default="gpt-4o-mini", help="model for frame descriptions")
    parser.add_argument("--selectmodel", type=str, default="gpt-4o", help="model for selecting moments")
    parser.add_argument("--folder_path", type=str, default="testfootage2", help="Path to the folder containing video files")
    parser.add_argument("--frame_extraction_frequency", type=int, default=64, help="Frequency of frame extraction (default: 96 frames)")
    return parser.parse_args()


# Main function
if __name__ == "__main__":
    args = parse_args()

    extractor = VideoMomentExtractor(args)
    extractor.run(args.folder_path)

    vis = FootageVisualizer("./")
    vis.visualize_footage(extractor.footage_data)