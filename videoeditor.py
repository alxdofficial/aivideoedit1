
import argparse
import numpy as np
from visualizeselection import FootageVisualizer
from momentselection import VideoMomentExtractor
from momentselectiononeshot import VideoMomentExtractorOneShot
from sequencer import Sequencer

# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using LLaVA or GPT-4V")
    # parser.add_argument("--visionmodel", type=str, default="llava", help="model for frame descriptions")
    parser.add_argument("--visionmodel", type=str, default="gpt-4o-mini", help="model for frame descriptions")
    parser.add_argument("--selectmodel", type=str, default="gpt-4o", help="model for selecting moments")
    parser.add_argument("--folder_path", type=str, default="testfootage1", help="Path to the folder containing video files")
    parser.add_argument("--frame_extraction_frequency", type=int, default=96, help="Frequency of frame extraction (default: 24 frames) 1 sec")
    # parser.add_argument("--user_prompt", type=str, default="create a energetic video of the lakers team in attack", help="prompt for the video")
    parser.add_argument("--user_prompt", type=str, default="create a scenic hiking video", help="prompt for the video")


    return parser.parse_args()


# Main function
if __name__ == "__main__":
    args = parse_args()

    # extractor = VideoMomentExtractor(args)
    # extractor.run(args.folder_path)

    extractor = VideoMomentExtractorOneShot(args)
    extractor.run(args.folder_path)

    vis = FootageVisualizer("./")
    vis.visualize_footage(extractor.footage_data)

    sequencer = Sequencer(args.selectmodel)
    sequencer.run(extractor.footage_data)