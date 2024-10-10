import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import math

class FootageVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def truncate_text(self, text, max_words=14):
        """
        Truncate the text to a maximum number of words.
        Args:
            text (str): The full text description.
            max_words (int): The maximum number of words to keep.
        Returns:
            str: The truncated text.
        """
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + '...'
        return text

    def visualize_footage(self, footage_data):
        """
        Visualize the frames of all videos in one large plot, with descriptions underneath
        and a green border around selected frames. Displays 4 frames per row.
        Args:
            footage_data (FootageData): The footage data structure containing video and frame information.
        """
        total_frames = sum(len(video.frames) for video in footage_data.videos)  # Total number of frames in all videos
        frames_per_row = 2  # Number of frames per row
        total_rows = math.ceil(total_frames / frames_per_row)  # Calculate the number of rows

        # Create a figure to plot all frames
        fig, axes = plt.subplots(nrows=total_rows, ncols=frames_per_row, figsize=(16, total_rows * 4))

        # Flatten axes array to easily access individual subplots
        axes = axes.flatten()

        current_frame_idx = 0

        # Loop through all videos and frames
        for video_idx, video in enumerate(footage_data.videos):
            for frame_idx, frame in enumerate(video.frames):
                # Convert the image frame (numpy array) to PIL Image for visualization
                img_pil = Image.fromarray(frame.image)

                # Apply a thick green border if the frame is chosen
                if frame.chosen:
                    img_pil = ImageOps.expand(img_pil, border=20, fill='green')

                # Plot the image
                axes[current_frame_idx].imshow(img_pil)
                axes[current_frame_idx].axis('off')

                # Add the frame ID and description below the image
                frame_id = f"Frame ID: {frame.id}"  # Bold frame ID (Markdown not supported in matplotlib)
                description = self.truncate_text(frame.description if frame.description else "No description")

                # Set the title: frame ID on top, description below
                axes[current_frame_idx].set_title(f"{frame_id}\n{description}", fontsize=10, loc='left')

                current_frame_idx += 1

        # Hide any unused axes
        for idx in range(current_frame_idx, len(axes)):
            axes[idx].axis('off')

        # Adjust layout and save the plot for all videos
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, "all_videos_frames.png")
        plt.savefig(output_file)
        plt.close(fig)
        print(f"Saved visualization of all videos at {output_file}")