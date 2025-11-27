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
    Expects masks as a list of binary numpy arrays (uint8) sized to original frame.
    """
    if not masks:
        return
    
    color = (0, 255, 0) # Green
    
    for i, mask in enumerate(masks):
        try:
            # Ensure mask is uint8 binary (0 or 1/255)
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = (mask > 0).astype(np.uint8)
            
            # If mask size doesn't match frame, resize it (final safety net)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
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
                    overlay = frame.copy()
                    cv2.drawContours(overlay, contours, -1, color, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
        except Exception as e:
            print(f"[ERROR] Draw mask failed: {e}")

def process_sam3_outputs_optimized(outputs, target_size, confidence_threshold, mask_threshold=0.5):
    """
    Memory-optimized post-processing with Enhanced Mask Quality.
    1. Filter masks based on IoU scores FIRST.
    2. Resize passing masks.
    3. Apply dynamic thresholding.
    4. Apply Morphological Smoothing (Open/Close).
    5. Filter small noise contours.
    """
    try:
        # 1. Get logits and scores
        # [batch, num_masks, height, width] -> remove batch
        pred_masks = outputs.pred_masks.squeeze(0) 
        
        # Get scores for filtering
        if hasattr(outputs, 'iou_scores'):
            scores = outputs.iou_scores.squeeze(0).squeeze(-1) # [num_masks]
        else:
            # Fallback if no scores
            scores = torch.ones(pred_masks.shape[0], device=pred_masks.device)

        # 2. FILTER FIRST
        # Find indices of masks that pass the confidence threshold
        keep_indices = torch.where(scores > confidence_threshold)[0]
        
        if len(keep_indices) == 0:
            return []
            
        # Keep only good masks
        filtered_masks = pred_masks[keep_indices]
        
        # 3. Resize ONLY the good masks
        filtered_masks = filtered_masks.unsqueeze(0)
        
        # Resize to target size
        resized_masks = F.interpolate(
            filtered_masks,
            size=target_size,
            mode="bilinear", # Bilinear is faster, morphology will clean it up
            align_corners=False
        ).squeeze(0)
        
        # 4. Sigmoid and Dynamic Binarization
        probs = torch.sigmoid(resized_masks)
        binary_masks = (probs > mask_threshold).float() 
        
        # 5. Convert to Numpy for Morphology & Cleaning
        final_masks = []
        binary_masks_np = binary_masks.cpu().numpy() 
        
        # Morphological Kernel (5x5 allows for decent smoothing)
        kernel = np.ones((5,5), np.uint8)
        
        for mask in binary_masks_np:
            # Convert to uint8 for OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # A. Morphological Opening (Remove small noise)
            mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            
            # B. Morphological Closing (Fill small holes)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            
            # C. Contour Filtering (Remove detached small islands)
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a new empty mask to draw only valid contours
            final_clean_mask = np.zeros_like(mask_cleaned)
            has_valid_object = False
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200: # Minimum area threshold (adjustable)
                    cv2.drawContours(final_clean_mask, [cnt], -1, 255, -1)
                    has_valid_object = True
            
            if has_valid_object:
                final_masks.append(final_clean_mask)
            
        return final_masks

    except Exception as e:
        print(f"[ERROR] Optimized post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
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
    
    # Reduce input size to save VRAM (Max dimension)
    MAX_INPUT_SIZE = 1024 

    while True:
        rtsp_url = app_state.get("rtsp_url")
        uploaded_image_path = app_state.get("uploaded_image_path")
        model = app_state.get("model")
        processor = app_state.get("processor")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        frame = None
        
        # 1. Acquire Frame logic (same as before)
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
        mask_thresh = app_state.get("mask_threshold", 0.5)
        display_mode = app_state.get("display_mode", "segmentation")
        
        should_process = (model and processor and (prompt or point_prompt))
        if rtsp_url:
             should_process = should_process and (frame_counter % 5 == 0)

        if should_process:
            # Save original dimensions for drawing later
            orig_h, orig_w = frame.shape[:2]
            
            # Resize frame for inference to save VRAM
            h, w = orig_h, orig_w
            scale = 1.0
            if max(h, w) > MAX_INPUT_SIZE:
                scale = MAX_INPUT_SIZE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame

            image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            inputs = None
            try:
                # Prepare Inputs
                if prompt:
                    inputs = processor(
                        text=[prompt],
                        images=image_pil,
                        return_tensors="pt"
                    ).to(device)
                
                elif point_prompt:
                    # Adjust point coords to resized image
                    curr_w, curr_h = image_pil.size
                    abs_x = int(point_prompt["x"] * curr_w)
                    abs_y = int(point_prompt["y"] * curr_h)
                    inputs = processor(
                        images=image_pil,
                        input_points=[[[abs_x, abs_y]]],
                        return_tensors="pt"
                    ).to(device)

                # Run Inference
                if inputs:
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Optimized Post-Processing (Target size is ORIGINAL frame size)
                    final_masks = process_sam3_outputs_optimized(
                        outputs, 
                        target_size=(orig_h, orig_w), 
                        confidence_threshold=confidence,
                        mask_threshold=mask_thresh
                    )
                    count = len(final_masks)
                    
                    # Draw on ORIGINAL frame
                    draw_masks(frame, final_masks, display_mode)
                    
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                torch.cuda.empty_cache() # Clear memory on error

        # 3. Update Status & Broadcast
        status = "Approved" if count >= app_state["max_limit"] else "Waiting"
        status_color = "green" if status == "Approved" else "orange"
        
        # Use lower quality JPEG for stream to save bandwidth/cpu
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
        
        if rtsp_url:
            await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.1)