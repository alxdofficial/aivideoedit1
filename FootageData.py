import random
import string
import numpy as np

# Frame class to hold frame-specific information
class Frame:
    def __init__(self, image, frame_number, description=None, chosen=False):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.image = image  # Actual image frame (as numpy array)
        self.frame_number = frame_number  # Frame number in the video
        self.description = description  # Frame description
        self.chosen = chosen  # Boolean to indicate if the frame is selected for the final cut


# Video class to hold video-specific information, including an array of frames
class Video:
    def __init__(self, container, total_frames, fps):
        self.container = container  # PyAV container for video metadata
        self.total_frames = total_frames  # Total number of frames in the video
        self.fps = fps  # Frame rate of the video
        self.frames = []  # List to hold Frame objects

    def add_frame(self, image, frame_number):
        frame = Frame(image=image, frame_number=frame_number)
        self.frames.append(frame)


# FootageData class to manage multiple videos and provide metadata
class FootageData:
    def __init__(self, folder_path):
        self.folder_path = folder_path  # Path to the folder containing video files
        self.videos = []  # List to hold Video objects

    def add_video(self, video):
        self.videos.append(video)

    def num_videos(self):
        return len(self.videos)
