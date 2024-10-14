
import argparse
import numpy as np
from momentsvisualizer import FootageVisualizer
from sequencervisualizer import SequencerVisualizer

from ragenhancedmomentselection import RAGEnhancedMomentSelector
from sequencer import Sequencer
import openai
# Argument parsing setup
def parse_args():
    parser = argparse.ArgumentParser(description="Video Moment Extractor using LLaVA or GPT-4V")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="model for frame descriptions")
    parser.add_argument("--folder_path", type=str, default="testfootage3", help="Path to the folder containing video files")
    parser.add_argument("--keyframe_frequency", type=int, default=2400, help="fhjgvjhkbl")
    parser.add_argument("--user_prompt", type=str, default="the lakers team taking shots at the hoop, scoring, and closeups", help="prompt for the video")
    # parser.add_argument("--user_prompt", type=str, default="create a scenic hiking video", help="prompt for the video")
    # parser.add_argument("--user_prompt", type=str, default="create a cool travel video", help="prompt for the video")

    return parser.parse_args()


# Main function
if __name__ == "__main__":
    args = parse_args()

    extractor = RAGEnhancedMomentSelector(args)
    chosen_frames = extractor.run(args.folder_path)

    fv = FootageVisualizer(".")
    fv.visualize_footage(chosen_frames)

    sequencer = Sequencer(args.gpt_model)
    ordered_ids = sequencer.run(chosen_frames)

    sv = SequencerVisualizer()
    sv.create_final_video(ordered_ids, chosen_frames)


    extractor.terminate_session()