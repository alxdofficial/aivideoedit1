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

            # Process frames in batches 
            batch_size = 4
            for batch_start in range(0, len(video.frames), batch_size):
                batch_end = min(batch_start + batch_size, len(video.frames))
                batch_frames = video.frames[batch_start:batch_end]

                print(f"Processing frames {batch_start} to {batch_end - 1}")

                # Create a conversation prompt for 8 frames
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe briefly what is happening in each of the following 8 frames. Return each description on a new line."
                            }
                        ] + [{"type": "image"} for _ in batch_frames],  # One image for each frame
                    }
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

                # Extract images for batch processing
                images = [frame.image for frame in batch_frames]

                # Process the batch of frames using the LLaVA model
                inputs = self.processor(text=prompt, images=images, padding=True, return_tensors="pt").to(self.device)
                output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)

                # Decode the output description
                batch_descriptions = self.processor.decode(output[0][2:], skip_special_tokens=True)

                # Split the descriptions into individual lines (one for each frame)
                descriptions = batch_descriptions.split('\n')[1:]

                start = 0
                for i, description in enumerate(descriptions):
                    if not len(description):
                        start += 1
                descriptions = descriptions[start:]


                # Assign descriptions to the corresponding frames
                for i, frame in enumerate(batch_frames):
                    if i < len(descriptions):
                        description = descriptions[i].strip()
                        keyword = "ASSISTANT:"
                        if keyword in description:
                            description = description.split(keyword, 1)[1].strip()
                        frame.description = description
                        # print(description)


