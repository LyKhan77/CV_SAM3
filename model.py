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
                    # Draw ID background for readability
                    label = f"#{i+1}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - 20), (x + tw + 4, y), color, -1)
                    cv2.putText(frame, label, (x + 2, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # Segmentation mode
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # SMOOTHING: Approximation
                    smoothed_contours = []
                    for cnt in contours:
                        epsilon = 0.005 * cv2.arcLength(cnt, True) # 0.5% error margin
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        smoothed_contours.append(approx)

                    cv2.drawContours(frame, smoothed_contours, -1, color, 2) # Draw smooth lines
                    
                    # Overlay fill
                    overlay = frame.copy()
                    cv2.drawContours(overlay, smoothed_contours, -1, color, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
                    # Calculate center for ID
                    if len(smoothed_contours) > 0:
                        M = cv2.moments(smoothed_contours[0])
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            # Draw ID
                            label = f"#{i+1}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (cX - tw//2 - 2, cY - th//2 - 2), (cX + tw//2 + 2, cY + th//2 + 2), (0,255,0), -1) # Green bg
                            cv2.putText(frame, label, (cX - tw//2, cY + th//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
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

        # 6. Aggressive Mask Merging (Glue Logic)
        # Combine fragmented masks (puzzle pieces) into single objects
        return merge_overlapping_masks(final_masks)

    except Exception as e:
        print(f"[ERROR] Optimized post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return []

def merge_overlapping_masks(masks):
    """
    Aggressively merges masks that overlap or touch.
    Useful for fixing fragmentation where one object is split into many parts.
    """
    if not masks:
        return []
        
    # 1. Create a graph of connected masks
    # Each mask is a node. Edge exists if masks overlap/touch.
    n = len(masks)
    parent = list(range(n))
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Pre-dilate masks slightly to bridge small gaps
    dilated_masks = []
    kernel = np.ones((9,9), np.uint8) # 9x9 dilation to connect nearby pieces
    for m in masks:
        dilated_masks.append(cv2.dilate(m, kernel, iterations=1))

    # Check for overlaps (O(N^2) but N is usually small < 100)
    for i in range(n):
        for j in range(i + 1, n):
            # Check intersection
            intersection = np.logical_and(dilated_masks[i], dilated_masks[j]).sum()
            if intersection > 0: # Any touch after dilation triggers merge
                union(i, j)
                
    # 2. Group masks by parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(masks[i])
        
    # 3. Fuse groups into single masks
    merged_masks = []
    for root in groups:
        group_masks = groups[root]
        if not group_masks: continue
            
        # Start with first mask
        fused_mask = group_masks[0].copy()
        for m in group_masks[1:]:
            fused_mask = cv2.bitwise_or(fused_mask, m)
            
        merged_masks.append(fused_mask)
        
    return merged_masks

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
    
    # Track active RTSP URL to detect changes
    current_active_rtsp = None
    
    # Reduce input size to save VRAM (Max dimension)
    MAX_INPUT_SIZE = 1024 
    
    # State Cache
    last_state_hash = None
    last_payload = None

    while True:
        # Get all state variables including video-related ones
        input_mode = app_state.get("input_mode", "rtsp")
        rtsp_url = app_state.get("rtsp_url")
        uploaded_image_path = app_state.get("uploaded_image_path")
        video_file_path = app_state.get("video_file_path")
        video_playing = app_state.get("video_playing", True)
        video_current_frame = app_state.get("video_current_frame", 0)
        video_seek_request = app_state.get("video_seek_request")
        model = app_state.get("model")
        processor = app_state.get("processor")
        prompt = app_state.get("prompt")
        point_prompt = app_state.get("point_prompt")
        confidence = app_state.get("confidence_threshold", 0.5)
        mask_thresh = app_state.get("mask_threshold", 0.5)
        display_mode = app_state.get("display_mode", "segmentation")
        sound_enabled = app_state.get("sound_enabled")
        max_limit = app_state.get("max_limit")

        # RTSP State Management
        # If URL changed or became empty, release resource
        if rtsp_url != current_active_rtsp:
            if video_capture and input_mode == "rtsp": # Only affect RTSP capture
                 print(f"--- RTSP URL changed/cleared. Releasing capture. Old: {current_active_rtsp}, New: {rtsp_url} ---")
                 video_capture.release()
                 video_capture = None
            current_active_rtsp = rtsp_url

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate State Hash (Tuple of all factors affecting output)
        # For RTSP and Video, we add frame info to force update. For Image, we don't.
        current_hash = (
            input_mode,
            uploaded_image_path,
            rtsp_url,
            video_file_path,
            video_current_frame,
            prompt,
            json.dumps(point_prompt) if point_prompt else None,
            confidence,
            mask_thresh,
            display_mode,
            sound_enabled,
            max_limit
        )

        # RTSP and Video always change
        is_dynamic = input_mode in ["rtsp", "video"]
        
        # Check Cache (Only for Static Images)
        # Force update if prompts are None but last payload had detections (Clearing state)
        force_update = False
        if not is_dynamic and last_payload:
            last_analytics = last_payload.get("analytics", {})
            was_detecting = last_analytics.get("detected_object") not in [None, "", "N/A"]
            is_cleared = (prompt is None or prompt == "") and (point_prompt is None)
            if was_detecting and is_cleared:
                force_update = True
                print("--- Force updating to clear mask ---")

        if not is_dynamic and not force_update and current_hash == last_state_hash and last_payload:
            # Just broadcast the cached result to keep UI alive
            await manager.broadcast(json.dumps(last_payload))
            await asyncio.sleep(0.1)
            continue

        frame = None
        
        # 1. Acquire Frame logic based on input mode
        frame_metadata = {}

        if input_mode == "image" and uploaded_image_path and not rtsp_url and not video_file_path:
            if video_capture: video_capture.release(); video_capture = None
            frame = cv2.imread(uploaded_image_path)
            if frame is None:
                await asyncio.sleep(1)
                continue
        elif input_mode == "rtsp" and rtsp_url and not uploaded_image_path and not video_file_path:
            if video_capture is None:
                try:
                    # Support device index for local webcam (e.g., '0', '1', etc.)
                    if rtsp_url.isdigit():
                        device_index = int(rtsp_url)
                        print(f"--- Opening local device {device_index} ---")
                        video_capture = cv2.VideoCapture(device_index)
                    else:
                        # RTSP URL stream
                        video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

                    if not video_capture.isOpened():
                        raise ValueError(f"Failed to open: {rtsp_url}")

                    if rtsp_url.isdigit():
                        print(f"--- Local Device {device_index} Connected ---")
                    else:
                        print("--- RTSP Connected ---")
                except Exception as e:
                    print(f"Error opening video source: {e}")
                    video_capture = None
                    await asyncio.sleep(2)
                    continue
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release(); video_capture = None
                continue
        elif input_mode == "video" and video_file_path and not rtsp_url and not uploaded_image_path:
            # Initialize video capture if needed
            if video_capture is None or not video_capture.isOpened():
                video_capture = cv2.VideoCapture(video_file_path)
                if not video_capture.isOpened():
                    print(f"Failed to open video file: {video_file_path}")
                    video_capture = None
                    await asyncio.sleep(1)
                    continue
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                app_state["video_total_frames"] = total_frames
                app_state["video_fps"] = fps
                print(f"Video loaded: {total_frames} frames, {fps:.2f} FPS")

            # Handle seek requests
            if video_seek_request is not None:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_seek_request)
                app_state["video_current_frame"] = video_seek_request
                app_state["video_seek_request"] = None

            # Read frame based on playback state
            if video_playing:
                ret, frame = video_capture.read()
                if not ret:
                    # Loop video
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    app_state["video_current_frame"] = 0
                    ret, frame = video_capture.read()
                    if ret:
                        app_state["video_current_frame"] = 1
                else:
                    app_state["video_current_frame"] += 1
            else:
                # Pause: get current frame
                current_frame_pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = video_capture.read()
                # Reset position if we accidentally advanced
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

            # Add video metadata for WebSocket
            frame_metadata = {
                "input_mode": "video",
                "video_current_frame": app_state["video_current_frame"],
                "video_total_frames": app_state.get("video_total_frames", 0),
                "video_fps": app_state.get("video_fps", 0),
                "video_playing": video_playing
            }
        else:
            await asyncio.sleep(0.5)
            continue

        # 2. Process Frame
        frame_counter += 1
        count = 0
        
        should_process = (model and processor and (prompt or point_prompt))
        if input_mode in ["rtsp", "video"]:
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
        
        # Merge analytics with any frame metadata
        analytics = {
            "input_mode": input_mode,
            "detected_object": prompt if prompt else ("Selected Object" if point_prompt else "N/A"),
            "count": count,
            "max_limit": app_state["max_limit"],
            "status": status,
            "status_color": status_color,
            "trigger_sound": (status == "Approved" and app_state["sound_enabled"])
        }

        # Add video metadata if available
        if frame_metadata:
            analytics.update(frame_metadata)

        payload = {
            "video_frame": f"data:image/jpeg;base64,{jpg_b64}",
            "analytics": analytics
        }

        # Update Cache
        if not is_dynamic:
            last_state_hash = current_hash
            last_payload = payload

        await manager.broadcast(json.dumps(payload))

        # Adjust sleep based on input mode
        if input_mode == "rtsp":
            await asyncio.sleep(0.01)
        elif input_mode == "video":
            # Control playback speed based on video FPS if available
            fps = app_state.get("video_fps", 30)
            await asyncio.sleep(1.0 / fps)
        else:  # image
            await asyncio.sleep(0.1)