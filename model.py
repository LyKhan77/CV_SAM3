import asyncio
import cv2
import base64
import json
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3ForVideoSegmentation

def load_model():
    """
    Loads the SAM-3 model and processor from Hugging Face.
    """
    try:
        print("--- Loading SAM-3 Model and Processor ---")
        # Determine the device to run the model on
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Specify the model ID from Hugging Face
        model_id = "facebook/sam3-huge-iou"
        
        # Load the processor and the model
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3ForVideoSegmentation.from_pretrained(model_id).to(device)
        
        print("--- Model and Processor Loaded Successfully ---")
        return model, processor
    except Exception as e:
        print(f"FATAL: Failed to load AI model: {e}")
        print("Please check your internet connection and Hugging Face token permissions.")
        return None, None

async def video_processing_loop(manager, app_state):
    """
    Main background task to connect to RTSP, read frames, perform AI inference, and broadcast results.
    """
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0
    # Process every Nth frame to manage performance. Adjust as needed.
    PROCESS_EVERY_N_FRAMES = 5 

    while True:
        rtsp_url = app_state.get("rtsp_url")
        model = app_state.get("model")
        processor = app_state.get("processor")

        # Connect/Reconnect logic
        if rtsp_url and video_capture is None:
            try:
                print(f"Attempting to connect to RTSP stream: {rtsp_url}")
                video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not video_capture.isOpened():
                    raise ValueError("Could not open video stream.")
                print("--- RTSP Stream Connected Successfully ---")
            except Exception as e:
                print(f"ERROR: Failed to connect to RTSP stream: {e}")
                video_capture, app_state["rtsp_url"] = None, None
                await asyncio.sleep(5)
                continue

        if not video_capture or not rtsp_url:
            if video_capture:
                video_capture.release()
                video_capture = None
            await asyncio.sleep(1)
            continue
            
        ret, frame = video_capture.read()
        if not ret:
            print("--- Could not read frame, disconnecting. ---")
            video_capture.release()
            video_capture, app_state["rtsp_url"] = None, None
            continue

        frame_counter += 1
        count = 0
        status = "Waiting"
        status_color = "orange"
        
        # Perform AI inference only if model is loaded, a prompt is set, and it's the Nth frame
        if model and processor and app_state.get("prompt") and (frame_counter % PROCESS_EVERY_N_FRAMES == 0):
            prompt = app_state.get("prompt")
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            try:
                # Prepare inputs for the model
                inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(model.device)
                
                # Perform inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Post-process to get segmentation masks
                processed_sizes = [image_pil.size[::-1]]
                masks = processor.post_process_for_video_segmentation(outputs, processed_sizes=processed_sizes)[0]
                
                # Count detected objects
                count = len(masks)
                
                # Draw masks on the frame
                for mask in masks:
                    # Convert boolean mask to a drawable contour
                    contour = mask.squeeze().cpu().numpy().astype(np.uint8)
                    # Find contours requires a binary image
                    contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            except Exception as e:
                print(f"ERROR during AI inference: {e}")

        # Update status based on count
        if count >= app_state["max_limit"]:
            status = "Approved"
            status_color = "green"
        
        # Encode frame to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        video_frame_b64 = f"data:image/jpeg;base64,{jpg_as_text}"

        # Construct and broadcast the data payload
        data_payload = {
            "timestamp": asyncio.get_event_loop().time(),
            "video_frame": video_frame_b64,
            "analytics": {
                "detected_object": app_state.get("prompt", "N/A"),
                "count": count,
                "max_limit": app_state.get("max_limit", 100),
                "status": status,
                "status_color": status_color,
                "trigger_sound": status == "Approved" and app_state.get("sound_enabled", False)
            }
        }
        
        await manager.broadcast(json.dumps(data_payload))
        await asyncio.sleep(0.05) # ~20 FPS broadcast rate