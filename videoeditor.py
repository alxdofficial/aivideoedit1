
import argparse
import numpy as np
from old_unused.visualizeselection import FootageVisualizer

from ragenhancedmomentselection import RAGEnhancedMomentSelector
from sequencer import Sequencer
import openai
# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using LLaVA or GPT-4V")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="model for frame descriptions")
    parser.add_argument("--folder_path", type=str, default="testfootage3", help="Path to the folder containing video files")
    parser.add_argument("--keyframe_frequency", type=int, default=1024, help="fhjgvjhkbl")
    parser.add_argument("--user_prompt", type=str, default="create a energetic video of the lakers team in attack", help="prompt for the video")
    # parser.add_argument("--user_prompt", type=str, default="create a scenic hiking video", help="prompt for the video")
    # parser.add_argument("--user_prompt", type=str, default="create a cool travel video", help="prompt for the video")

    return parser.parse_args()


# Main function
if __name__ == "__main__":
    args = parse_args()

    extractor = RAGEnhancedMomentSelector(args)
    chosen_frames = extractor.run(args.folder_path)

    sequencer = Sequencer(args.gpt_model)
    sequencer.run(chosen_frames)

    extractor.terminate_session()