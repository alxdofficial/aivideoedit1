class LlavaFootageComprehension:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def comprehend_frames(self, footage_data):
        '''
        Process the frames using LLaVA and set descriptions for each frame in the FootageData.
        Args:
            footage_data (FootageData): The footage data containing videos and frames.
        '''
        print(f"Comprehending frames in {footage_data.num_videos()} videos using LLaVA...")

        # Iterate over each video in the footage data
        for video in footage_data.videos:
            print(f"Processing video with {len(video.frames)} frames.")
            for frame in video.frames:
                print(f"Processing frame {frame.frame_number}")

                # Create a single frame conversation prompt
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe biefly what is happening in this frame."},
                            {"type": "image"}
                        ],
                    }
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

                # Process the frame using the LLaVA model
                inputs = self.processor(text=prompt, images=[frame.image], padding=True, return_tensors="pt").to(self.device)
                output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

                # Decode the output description
                description = self.processor.decode(output[0][2:], skip_special_tokens=True)

                keyword = "ASSISTANT:"
                if keyword in description:
                    description = description.split(keyword, 1)[1].strip()

                # Set the description field for the current frame
                frame.description = description
