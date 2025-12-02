import asyncio
import cv2
import base64
import json
import torch
import torchvision # Added for NMS
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import torch.nn.functional as F

def draw_masks(frame, masks):
    """
    Draw masks on frame (Always in Segmentation Mode).
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
                
                # Removed ID Text Overlay per request
                    
        except Exception as e:
            print(f"[ERROR] Draw mask failed: {e}")

def process_sam3_outputs_optimized(outputs, target_size, confidence_threshold, mask_threshold=0.5):
    """
    Memory-optimized post-processing with Enhanced Mask Quality & NMS.
    Returns: List of binary masks (numpy arrays)
    """
    try:
        # ... (rest of function remains same until return)
        # The function already returns binary masks (cleaned), which is what we want for 'raw' export
        # The smoothing happens in draw_masks, so this output IS the high fidelity one.
        return final_masks 

    except Exception as e:
        print(f"[ERROR] Optimized post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return []

# ... (merge_overlapping_masks and load_model remain same)

async def video_processing_loop(manager, app_state):
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0
    
    # Log Cache
    last_logged_count = -1
    
    # Default segment state
    app_state["should_segment"] = False # Default to False, wait for Run button

    while True:
        # Get all state variables
        input_mode = app_state.get("input_mode", "rtsp")
        # ... (other gets)
        should_segment = app_state.get("should_segment", False)

        # ... (RTSP/Hash logic remains)

        # 2. Process Frame
        frame_counter += 1
        count = 0
        final_masks = [] # Default empty
        
        # Only process if should_segment is TRUE
        should_process = (model and processor and (prompt or point_prompt) and should_segment)
        
        if input_mode in ["rtsp", "video"]:
             should_process = should_process and (frame_counter % 5 == 0)

        if should_process:
             # ... (Inference logic)
                    # Optimized Post-Processing (Returns high fidelity masks)
                    final_masks = process_sam3_outputs_optimized(
                        outputs, 
                        target_size=(orig_h, orig_w), 
                        confidence_threshold=confidence,
                        mask_threshold=mask_thresh
                    )
                    count = len(final_masks)
                    
                    # SAVE STATE FOR EXPORT (Raw Masks & Frame)
                    app_state["last_processed_frame"] = frame.copy()
                    app_state["last_raw_masks"] = final_masks 
                    
                    # Draw on ORIGINAL frame (Applies smoothing for UI)
                    draw_masks(frame, final_masks)
        
        # If NOT processing but we have cached result (e.g. static image after run), 
        # we might need to redraw the last known masks if we are just refreshing the frame?
        # For now, if should_segment is False (Clear Mask), we send clean frame.
        # If should_segment is True but no change, we resend last frame with masks.
        
        # ... (Status & Broadcast logic)
        
        # Update Status Text based on state
        process_status = "Processing..." if should_process else ("Done" if count > 0 else "Ready")
        if not should_segment: process_status = "Ready"

        analytics = {
            "input_mode": input_mode,
            "detected_object": prompt if prompt else "N/A",
            "process_status": process_status, # New field for UI
            "count": count,
            # ...
        }

def process_sam3_outputs_optimized(outputs, target_size, confidence_threshold, mask_threshold=0.5):
    """
    Memory-optimized post-processing with Enhanced Mask Quality & NMS.
    """
    try:
        # 1. Get logits and scores
        pred_masks = outputs.pred_masks.squeeze(0) 
        
        if hasattr(outputs, 'iou_scores'):
            scores = outputs.iou_scores.squeeze(0).squeeze(-1)
        else:
            scores = torch.ones(pred_masks.shape[0], device=pred_masks.device)

        # 2. FILTER BY SCORE FIRST
        keep_indices = torch.where(scores > confidence_threshold)[0]
        if len(keep_indices) == 0:
            return []
            
        filtered_masks = pred_masks[keep_indices]
        filtered_scores = scores[keep_indices]
        
        # 3. RESIZE
        filtered_masks = filtered_masks.unsqueeze(0)
        resized_masks = F.interpolate(
            filtered_masks,
            size=target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        # 4. NMS (Non-Maximum Suppression) to remove overlaps
        # Convert masks to boxes for NMS
        # Note: This is a heuristic. Ideally NMS is done on masks, but box NMS is faster and usually sufficient.
        binary_masks_raw = (resized_masks > mask_threshold).float()
        boxes = []
        valid_indices = []
        
        for i in range(binary_masks_raw.shape[0]):
            # Find bounding box of the mask
            mask_tensor = binary_masks_raw[i]
            y, x = torch.where(mask_tensor > 0)
            if len(x) > 0 and len(y) > 0:
                x1, x2 = x.min(), x.max()
                y1, y2 = y.min(), y.max()
                boxes.append([x1.float(), y1.float(), x2.float(), y2.float()])
                valid_indices.append(i)
        
        if not boxes:
            return []
            
        boxes_tensor = torch.stack([torch.tensor(b) for b in boxes]).to(pred_masks.device)
        scores_tensor = filtered_scores[valid_indices]
        
        # Apply NMS (IoU threshold 0.5 means if overlap > 50%, drop lower score)
        keep_nms = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.5)
        
        # Get final masks based on NMS indices
        final_indices = [valid_indices[k] for k in keep_nms]
        
        # 5. Convert to Numpy for Morphology
        final_masks = []
        # Use sigmoid on the specific indices that passed NMS
        probs = torch.sigmoid(resized_masks[final_indices])
        binary_masks = (probs > mask_threshold).float() 
        binary_masks_np = binary_masks.cpu().numpy() 
        
        kernel = np.ones((5,5), np.uint8)
        
        for mask in binary_masks_np:
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            final_clean_mask = np.zeros_like(mask_cleaned)
            has_valid_object = False
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200: 
                    cv2.drawContours(final_clean_mask, [cnt], -1, 255, -1)
                    has_valid_object = True
            
            if has_valid_object:
                final_masks.append(final_clean_mask)

        return final_masks # Skip merging logic if NMS is used, as NMS prevents fragmentation usually

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

    # Log Cache
    last_logged_count = -1

    # Video open retry tracking
    video_open_retry_count = 0
    MAX_VIDEO_OPEN_RETRIES = 3

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
        sound_enabled = app_state.get("sound_enabled")
        max_limit = app_state.get("max_limit")
        should_segment = app_state.get("should_segment", False)

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
            sound_enabled,
            max_limit,
            should_segment
        )

        # RTSP and Video always change
        is_dynamic = input_mode in ["rtsp", "video"]
        
        # Check Cache (Only for Static Images)
        # Force update if prompts are None but last payload had detections (Clearing state)
        force_update = False
        if not is_dynamic and last_payload:
            last_analytics = last_payload.get("analytics", {})
            was_prompt_set = last_analytics.get("detected_object") not in [None, "", "N/A"]
            was_processing = last_analytics.get("process_status") == "Done"
            
            is_cleared = (prompt is None or prompt == "") and (point_prompt is None)
            
            # Trigger 1: Prompt was removed
            cond_prompt_removed = (was_prompt_set and is_cleared)
            # Trigger 2: We were actively segmenting but now stopped (Clear Mask button)
            cond_stopped_segmenting = (was_processing and not should_segment)

            if cond_prompt_removed or cond_stopped_segmenting:
                force_update = True
                print("--- INFO: Clearing mask state... ---")
                last_payload = None # Fixes infinite loop
                print("--- INFO: Mask state cleared ---")

        if not is_dynamic and not force_update and current_hash == last_state_hash and last_payload:
            # Just broadcast the cached result to keep UI alive
            await manager.broadcast(json.dumps(last_payload))
            await asyncio.sleep(0.1)
            continue

        frame = None
        
        # 1. Acquire Frame logic based on input mode
        frame_metadata = {}

        if input_mode == "image":
            if not uploaded_image_path:
                 # Invalidate cache if image is gone
                 last_state_hash = None
                 last_payload = None
                 await asyncio.sleep(0.5)
                 continue

            if not rtsp_url and not video_file_path:
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
                # Check if already opened during upload
                existing_capture = app_state.get("video_capture")
                if existing_capture and existing_capture.isOpened():
                    video_capture = existing_capture
                    print("Reusing existing VideoCapture from upload")
                else:
                    video_capture = cv2.VideoCapture(video_file_path)
                if not video_capture.isOpened():
                    video_open_retry_count += 1
                    print(f"Failed to open video file: {video_file_path} (attempt {video_open_retry_count}/{MAX_VIDEO_OPEN_RETRIES})")

                    if video_open_retry_count >= MAX_VIDEO_OPEN_RETRIES:
                        # Send error notification to frontend via WebSocket
                        error_payload = {
                            "status": "error",
                            "message": f"Failed to open video after {MAX_VIDEO_OPEN_RETRIES} attempts",
                            "video_frame": None,
                            "analytics": {
                                "input_mode": "video",
                                "error": f"Video file inaccessible: {video_file_path}"
                            }
                        }
                        await manager.broadcast(json.dumps(error_payload))
                        video_open_retry_count = 0  # Reset counter
                        await asyncio.sleep(5)  # Wait longer before next attempt
                    else:
                        await asyncio.sleep(1)

                    video_capture = None
                    continue
                else:
                    video_open_retry_count = 0  # Reset on success

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
                        app_state["video_current_frame"] = 0  # Fixed: Was 1, now 0
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
        
        should_process = (model and processor and (prompt or point_prompt) and should_segment)
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
                    # Use ORIGINAL frame size for coordinate calculation
                    abs_x = int(point_prompt["x"] * orig_w)
                    abs_y = int(point_prompt["y"] * orig_h)

                    # Scale to resized image coordinates
                    curr_w, curr_h = image_pil.size
                    scaled_x = int(abs_x * (curr_w / orig_w))
                    scaled_y = int(abs_y * (curr_h / orig_h))

                    inputs = processor(
                        images=image_pil,
                        input_points=[[[scaled_x, scaled_y]]],
                        return_tensors="pt"
                    ).to(device)

                # Run Inference
                if inputs:
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # ✅ OFFICIAL SAM-3 POST-PROCESSING (Meta's optimized pipeline)
                    # Replaces custom process_sam3_outputs_optimized() for smooth, accurate results
                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=confidence,  # Confidence threshold (IoU-based filtering)
                        mask_threshold=mask_thresh,  # Mask binarization threshold
                        target_sizes=[[orig_h, orig_w]]  # Resize masks to original frame size
                    )[0]

                    # Extract and convert masks to numpy uint8 format (for drawing & ObjectList/ export)
                    final_masks = []
                    if 'masks' in results and len(results['masks']) > 0:
                        for mask in results['masks']:
                            # Convert tensor/bool to numpy uint8
                            if isinstance(mask, torch.Tensor):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)

                            # Ensure uint8 format (0-255) for compatibility with draw_masks() and export
                            if mask_np.dtype == bool:
                                mask_uint8 = (mask_np.astype(np.uint8)) * 255
                            elif mask_np.max() <= 1.0:
                                mask_uint8 = (mask_np * 255).astype(np.uint8)
                            else:
                                mask_uint8 = mask_np.astype(np.uint8)

                            final_masks.append(mask_uint8)

                    count = len(final_masks)

                    # SAVE STATE FOR EXPORT (Raw Masks & Frame)
                    app_state["last_processed_frame"] = frame.copy()
                    app_state["last_raw_masks"] = final_masks  # Compatible format for ObjectList/ export

                    # Draw on ORIGINAL frame
                    draw_masks(frame, final_masks)
                    
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                torch.cuda.empty_cache() # Clear memory on error

        # Update Status & Broadcast
        status = "Approved" if count >= app_state["max_limit"] else "Waiting"
        status_color = "green" if status == "Approved" else "orange"
        
        # WARNING LOGIC (Backend-driven Toast)
        # Reset warning flag if we find objects or stop segmenting
        if count > 0 or not should_segment:
            app_state["warning_sent"] = False
            
        warning_msg = None
        if should_segment and count == 0 and not app_state.get("warning_sent", False):
             warning_msg = "No objects detected. Try lowering confidence."
             app_state["warning_sent"] = True
             print(f"INFO: Inference finished but 0 objects. Sending UI Warning.")

        # LOGGING: Detect Success (State Change)
        if count > 0 and count != last_logged_count:
            print(f"INFO: Detecting Successful: Found {count} objects")
            last_logged_count = count
        elif count == 0:
            if should_process and last_logged_count != 0:
                print(f"INFO: Inference finished but 0 objects passed threshold (Conf: {confidence:.2f}, Mask: {mask_thresh:.2f}).")
            last_logged_count = 0 # Reset if empty
        
        # Use lower quality JPEG for stream to save bandwidth/cpu
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        jpg_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Determine Process Status for UI (Unlock "Processing..." state)
        process_status = "Done" if app_state.get("should_segment") else "Ready"

        # Merge analytics with any frame metadata
        analytics = {
            "input_mode": input_mode,
            "detected_object": prompt if prompt else ("Selected Object" if point_prompt else "N/A"),
            "count": count,
            "max_limit": app_state["max_limit"],
            "status": status,
            "status_color": status_color,
            "process_status": process_status,
            "warning": warning_msg, # Send warning to UI
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

            # Validate FPS value to prevent division by zero or invalid sleep
            if fps <= 0 or fps > 240:  # Reasonable FPS range: 1-240
                fps = 30  # Fallback to default

            await asyncio.sleep(1.0 / fps)
        else:  # image
            await asyncio.sleep(0.1)