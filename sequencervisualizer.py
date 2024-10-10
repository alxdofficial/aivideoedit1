import os
import av
import cv2
import numpy as np

class SequencerVisualizer:
    def __init__(self, output_dir="output_videos"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_clip(self, video_path, frame_number, fps, clip_duration=4, output_size=(1920, 1080)):
        '''
        Extract a clip of the video, starting 2 seconds before the selected frame and ending 2 seconds after.
        Args:
            video_path (str): Path to the video file.
            frame_number (int): Frame number of the selected clip.
            fps (int): Frames per second of the video.
            clip_duration (int): Duration of the clip in seconds (default 4 seconds).
            output_size (tuple): Size to resize each frame (default 1920x1080).
        Returns:
            List of frames (as numpy arrays) for the extracted clip.
        '''
        # Calculate start and end frames for the clip, ensuring values are integers
        start_frame = max(0, int(frame_number - fps * 2))  # 2 seconds before the selected frame
        end_frame = int(frame_number + fps * 2)  # 2 seconds after the selected frame
        
        # Open the video file using PyAV
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Initialize an array to store the frames
        frames = []
        current_frame = 0

        # Seek to the start frame and decode frames
        container.seek(int(start_frame), stream=stream)
        for frame in container.decode(stream):
            if start_frame <= current_frame <= end_frame:
                img = frame.to_ndarray(format="rgb24")
                
                # Resize the frame to the desired output size (1920x1080)
                img_resized = cv2.resize(img, output_size)
                frames.append(img_resized)

            # Stop when we reach the end frame
            if current_frame > end_frame:
                break

            current_frame += 1
        
        return frames

    def combine_clips(self, clip_frames, output_path, output_size=(1920, 1080)):
        '''
        Combine the extracted clips and save them as a video file.
        Args:
            clip_frames (list): List of all frames for all clips.
            output_path (str): Path to save the combined video.
            output_size (tuple): Size of the output video (default 1920x1080).
        '''
        if len(clip_frames) == 0:
            print("No clips to combine.")
            return
        
        # Determine frame size from the output size
        frame_width, frame_height = output_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 video
        
        # Initialize the video writer
        video_writer = cv2.VideoWriter(output_path, fourcc, 24, (frame_width, frame_height))
        
        # Write each frame of every clip
        for frames in clip_frames:
            for frame in frames:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Combined video saved at {output_path}")

    def create_final_video(self, ordered_clips, chosen_frames, output_size=(1920, 1080)):
        '''
        Create a combined video from the selected ordered clips.
        Args:
            ordered_clips (list): List of ordered clip IDs.
            frames (list): List of selected Frame objects.
            output_size (tuple): Size to resize the frames (default 1920x1080).
        '''
        output_frames = []
        for clip_id in ordered_clips:
            # Find the corresponding frame and video path
            for frame in chosen_frames:
                if frame.id == clip_id:
                    video_path = frame.video_path
                    # Extract frames for this clip
                    frames = self.extract_clip(video_path, frame.frame_number, fps=24, output_size=output_size)  # Assuming fps=24
                    output_frames.append(frames)
                    break

        # Combine the clips and save the video
        output_path = os.path.join(self.output_dir, "sequencer_visualization.mp4")
        self.combine_clips(output_frames, output_path, output_size=output_size)
