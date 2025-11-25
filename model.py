import asyncio
import cv2
import base64
import json
import torch
import numpy as np
from PIL import Image
# Correct imports based on user-provided documentation
from transformers import AutoProcessor, AutoModel

def load_model():
    """
    Loads the SAM-3 model and processor from Hugging Face using the correct identifier.
    """
    try:
        print("--- Loading SAM-3 Model and Processor ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Correct model identifier from user-provided documentation
        model_id = "facebook/sam3"
        
        # Load the processor and the model using Auto-classes
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        
        print("--- Model and Processor Loaded Successfully ---")
        return model, processor
    except Exception as e:
        print(f"FATAL: Failed to load AI model: {e}")
        print("Please check your internet connection and that you have accepted the model's terms on Hugging Face.")
        return None, None

async def video_processing_loop(manager, app_state):
    """
    Main background task to connect to RTSP, read frames, perform AI inference, and broadcast results.
    """
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 10  # Adjust for performance

    while True:
        rtsp_url = app_state.get("rtsp_url")
        model = app_state.get("model")
        processor = app_state.get("processor")

        if not rtsp_url:
            if video_capture: video_capture.release(); video_capture = None
            await asyncio.sleep(1)
            continue

        if video_capture is None:
            try:
                print(f"Attempting to connect to RTSP stream: {rtsp_url}")
                video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not video_capture.isOpened(): raise ValueError("Could not open stream.")
                print("--- RTSP Stream Connected ---")
            except Exception as e:
                print(f"ERROR: Failed to connect to RTSP stream: {e}")
                video_capture, app_state["rtsp_url"] = None, None
                await asyncio.sleep(5)
                continue
            
        ret, frame = video_capture.read()
        if not ret:
            print("--- Could not read frame, disconnecting. ---")
            video_capture.release(); video_capture, app_state["rtsp_url"] = None, None
            continue

        frame_counter += 1
        count = 0
        prompt = app_state.get("prompt")
        
        # Only perform inference if all components are ready
        if model and processor and prompt and (frame_counter % PROCESS_EVERY_N_FRAMES == 0):
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                # 1. Process image and text inputs separately
                image_inputs = processor(images=image_pil, return_tensors="pt")
                text_inputs = processor(text=prompt, return_tensors="pt")
                
                # 2. Manually combine the inputs
                inputs = {
                    "pixel_values": image_inputs.pixel_values,
                    "input_ids": text_inputs.input_ids
                }
                
                # 3. Move all tensor inputs to the correct device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Post-process to get masks, providing the original image size.
                original_size = image_pil.size[::-1] # (height, width)
                masks = processor.post_process_segmentation(
                    outputs, 
                    original_sizes=[original_size], 
                    target_sizes=[original_size]
                )[0]
                count = len(masks)
                
                # Draw masks
                for mask in masks.cpu().numpy():
                    contour_img = (mask > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            except Exception as e:
                print(f"ERROR during AI inference: {e}")

        # Update status and encode frame
        status = "Approved" if count >= app_state["max_limit"] else "Waiting"
        status_color = "green" if status == "Approved" else "orange"
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Broadcast payload
        data_payload = {
            "video_frame": f"data:image/jpeg;base64,{jpg_as_text}",
            "analytics": {
                "detected_object": prompt or "N/A", "count": count,
                "max_limit": app_state["max_limit"], "status": status,
                "status_color": status_color,
                "trigger_sound": status == "Approved" and app_state["sound_enabled"]
            }
        }
        await manager.broadcast(json.dumps(data_payload))
        await asyncio.sleep(0.05)