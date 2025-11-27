import asyncio
import cv2
import base64
import json
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import torch.nn.functional as F

def draw_masks(frame, masks, display_mode):
    """
    Draw masks on frame based on display mode.
    Expects masks as a list of binary numpy arrays (uint8).
    """
    if not masks:
        return
    
    # Colors for bounding boxes/masks
    color = (0, 255, 0) # Green
    
    for i, mask in enumerate(masks):
        try:
            # Ensure mask is uint8 binary (0 or 1/255)
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = (mask > 0).astype(np.uint8)
                
            if display_mode == "bounding_box":
                x, y, w, h = cv2.boundingRect(mask)
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Obj {i+1}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Segmentation mode
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(frame, contours, -1, color, 2)
                    # Add semi-transparent fill
                    overlay = frame.copy()
                    cv2.drawContours(overlay, contours, -1, color, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
        except Exception as e:
            print(f"[ERROR] Draw mask failed: {e}")

def process_sam3_outputs_simple(outputs, target_size, confidence_threshold):
    """
    Simplified post-processing: manually resize and threshold logits.
    target_size: (height, width)
    """
    try:
        # 1. Get logits
        # Output shape usually: [batch, num_masks, height, width]
        pred_masks = outputs.pred_masks
        
        # Remove batch dim if present
        if len(pred_masks.shape) == 4:
            pred_masks = pred_masks.squeeze(0)
            
        # 2. Resize masks to original image size
        # Add batch and channel dims for interpolation: [1, 1, H, W] -> [num_masks, 1, H, W]
        # We iterate or do batch resize. Batch is faster.
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1) # [num_masks, 1, H, W]
            
        resized_masks = F.interpolate(
            pred_masks,
            size=target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1) # Back to [num_masks, H, W]
        
        # 3. Apply sigmoid and threshold
        probs = torch.sigmoid(resized_masks)
        binary_masks = (probs > confidence_threshold).float()
        
        # 4. Filter and convert to numpy list
        final_masks = []
        
        # Get IoU scores if available for filtering
        if hasattr(outputs, 'iou_scores'):
            scores = outputs.iou_scores.squeeze().cpu().numpy()
            if scores.ndim == 0: scores = [scores] # Handle single mask case
        else:
            scores = [1.0] * len(binary_masks)
            
        binary_masks_np = binary_masks.cpu().numpy()
        
        for i, mask in enumerate(binary_masks_np):
            # Filter by score/confidence
            if i < len(scores) and scores[i] < confidence_threshold:
                continue
                
            # Filter empty masks
            if mask.sum() < 10:
                continue
                
            # Convert to uint8 (0-255) for OpenCV
            final_masks.append((mask * 255).astype(np.uint8))
            
        return final_masks

    except Exception as e:
        print(f"[ERROR] Simple post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_model():
    try:
        print("--- Loading SAM-3 Model ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model_id = "facebook/sam3"
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id).to(device)
        
        print("--- Model Loaded Successfully ---")
        return model, processor
    except Exception as e:
        print(f"FATAL: Failed to load model: {e}")
        return None, None

async def video_processing_loop(manager, app_state):
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 1 # Can be increased for RTSP to reduce lag

    while True:
        rtsp_url = app_state.get("rtsp_url")
        uploaded_image_path = app_state.get("uploaded_image_path")
        model = app_state.get("model")
        processor = app_state.get("processor")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        frame = None
        
        # 1. Acquire Frame
        if uploaded_image_path and not rtsp_url:
            if video_capture: video_capture.release(); video_capture = None
            frame = cv2.imread(uploaded_image_path)
            if frame is None: 
                await asyncio.sleep(1)
                continue
                
        elif rtsp_url and not uploaded_image_path:
            if video_capture is None:
                try:
                    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if not video_capture.isOpened(): raise ValueError("Open failed")
                    print("--- RTSP Connected ---")
                except:
                    video_capture = None
                    await asyncio.sleep(2)
                    continue
            
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release(); video_capture = None
                continue
        else:
            await asyncio.sleep(0.5)
            continue

        # 2. Process Frame
        frame_counter += 1
        count = 0
        prompt = app_state.get("prompt")
        point_prompt = app_state.get("point_prompt")
        confidence = app_state.get("confidence_threshold", 0.5)
        display_mode = app_state.get("display_mode", "segmentation")
        
        should_process = (model and processor and (prompt or point_prompt))
        if rtsp_url: # Skip frames for RTSP to keep up
             should_process = should_process and (frame_counter % 5 == 0)

        if should_process:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target_size = (image_pil.height, image_pil.width)
            inputs = None
            
            try:
                # Prepare Inputs
                if prompt:
                    # Text Prompt
                    inputs = processor(
                        text=[prompt],
                        images=image_pil,
                        return_tensors="pt"
                    ).to(device)
                
                elif point_prompt:
                    # Point Prompt
                    w, h = image_pil.size
                    abs_x = int(point_prompt["x"] * w)
                    abs_y = int(point_prompt["y"] * h)
                    inputs = processor(
                        images=image_pil,
                        input_points=[[[abs_x, abs_y]]], # Batch size 1, 1 point
                        return_tensors="pt"
                    ).to(device)

                # Run Inference
                if inputs:
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Process Outputs
                    final_masks = process_sam3_outputs_simple(outputs, target_size, confidence)
                    count = len(final_masks)
                    
                    # Draw
                    draw_masks(frame, final_masks, display_mode)
                    
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")

        # 3. Update Status & Broadcast
        status = "Approved" if count >= app_state["max_limit"] else "Waiting"
        status_color = "green" if status == "Approved" else "orange"
        
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_b64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "video_frame": f"data:image/jpeg;base64,{jpg_b64}",
            "analytics": {
                "detected_object": prompt if prompt else ("Selected Object" if point_prompt else "N/A"),
                "count": count,
                "max_limit": app_state["max_limit"],
                "status": status,
                "status_color": status_color,
                "trigger_sound": (status == "Approved" and app_state["sound_enabled"])
            }
        }
        
        await manager.broadcast(json.dumps(payload))
        
        # Control loop speed
        if rtsp_url:
            await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.1) # Slower for static image to save CPU
