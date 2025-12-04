import asyncio
import cv2
import base64
import json
import os
import torch
import torchvision # Added for NMS
import numpy as np
import colorsys
from PIL import Image
from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel
import torch.nn.functional as F

def draw_masks(frame, masks):
    """
    Draw masks on frame (Always in Segmentation Mode).
    Expects masks as a list of binary numpy arrays (uint8) sized to original frame.
    """
    if not masks:
        return
    
    for i, mask in enumerate(masks):
        # Generate unique color for each mask using Golden Angle approach
        # This ensures distinct colors even for adjacent indices
        hue = (i * 0.618033988749895) % 1.0
        # High saturation (0.85) and value (0.95) for bright, visible colors
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        # Convert to BGR (0-255) for OpenCV
        color = (int(b * 255), int(g * 255), int(r * 255))
        
        try:
            # Ensure mask is uint8 0-255 range for resizing and blurring
            # If mask is 0/1, scale to 0/255
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            # If mask size doesn't match frame, resize it with LINEAR interpolation for smoothness
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Apply slight blur to soften pixelated edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
            # Segmentation mode
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # SMOOTHING: Approximation
                smoothed_contours = []
                for cnt in contours:
                    # Use very small epsilon (0.2%) to keep details but remove pixel steps
                    epsilon = 0.002 * cv2.arcLength(cnt, True) 
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
        print("\n" + "="*60)
        print("SAM 3 MODEL SELECTION")
        print("="*60)
        print("\n[1] Sam3Model - Text Prompts (Concept Segmentation)")
        print("    ✓ Supports: Text prompts (e.g., 'cat', 'person')")
        print("    ✗ Does NOT support: Point/Click prompts")
        print("    Use for: Image/Video mode with text input\n")

        print("[2] Sam3TrackerModel - Point/Click Prompts (RECOMMENDED)")
        print("    ✓ Supports: Point prompts, Box prompts")
        print("    ✗ Does NOT support: Text prompts")
        print("    Use for: Interactive Segmentation with clicks")
        print("    ⭐ Best choice for this app\n")
        print("="*60)

        choice = input("\nSelect model [1/2] (default: 2): ").strip()

        if choice == "1":
            model_type = "text"
            model_id = "facebook/sam3"
            print("\n✓ Loading Sam3Model (Text Prompts)...")
        else:  # Default to TrackerModel (choice == "2" or empty)
            model_type = "tracker"
            model_id = "facebook/sam3"  # Same model ID, different class
            print("\n✓ Loading Sam3TrackerModel (Point Prompts)...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if model_type == "text":
            processor = Sam3Processor.from_pretrained(model_id)
            model = Sam3Model.from_pretrained(model_id).to(device)
        else:  # tracker
            processor = Sam3TrackerProcessor.from_pretrained(model_id)
            model = Sam3TrackerModel.from_pretrained(model_id).to(device)

        print("--- Model Loaded Successfully ---\n")

        # Store model type in model object for later reference
        model.model_type = model_type

        return model, processor
    except Exception as e:
        print(f"FATAL: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

async def batch_process_video(app_state, video_path, prompt, point_prompt, confidence, mask_thresh, max_input_size=1024):
    """
    Batch process all frames in a video file with progress tracking.
    Stores processed frames (with masks drawn) in app_state["video_cache"].

    Args:
        app_state: Application state dictionary
        video_path: Path to video file
        prompt: Text prompt for segmentation
        point_prompt: Point coordinates for segmentation
        confidence: Confidence threshold
        mask_thresh: Mask threshold
        max_input_size: Max input resolution for inference

    Returns:
        True if successful, False if error/cancelled
    """
    print(f"--- Starting Batch Processing: {video_path} ---")

    # Close preview capture if exists (cleanup from raw playback mode)
    if hasattr(app_state, '_preview_cap') and app_state.get('_preview_cap') is not None:
        app_state['_preview_cap'].release()
        app_state['_preview_cap'] = None
        app_state['_preview_video_path'] = None

    # 1. Open video and get metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Limit processing to max 1000 frames for memory safety
    MAX_FRAMES = 1000
    if total_frames > MAX_FRAMES:
        print(f"[WARNING] Video has {total_frames} frames, limiting to {MAX_FRAMES}")
        total_frames = MAX_FRAMES

    print(f"Batch processing {total_frames} frames at {fps:.2f} FPS")

    # Setup Video Writer
    os.makedirs("processed_videos", exist_ok=True)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_segmented.mp4"
    output_path = os.path.join("processed_videos", output_filename)
    
    # Get properties for writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize VideoWriter
    # mp4v is widely supported for .mp4 container
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"[INFO] Recording processed video to: {output_path}")

    # 2. Initialize cache
    app_state["video_cache"] = {
        "video_path": video_path,
        "prompt": prompt,
        "confidence": confidence,
        "mask_threshold": mask_thresh,
        "total_frames": total_frames,
        "fps": fps,
        "frames": [],
        "processing_complete": False,
        "current_progress": 0,
        "output_path": output_path, # Store path for download
    }

    # 3. Set batch processing flag
    app_state["batch_processing_active"] = True
    app_state["batch_progress_total"] = total_frames

    # Get model references
    model = app_state.get("model")
    processor = app_state.get("processor")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not model or not processor:
        print("[ERROR] Model not loaded!")
        app_state["batch_processing_active"] = False
        cap.release()
        if out_writer: out_writer.release()
        return False

    # 4. Process each frame
    processed_frames = []

    for frame_idx in range(total_frames):
        # Check if processing was cancelled
        if not app_state.get("batch_processing_active", False):
            print("[INFO] Batch processing cancelled by user")
            cap.release()
            if out_writer: out_writer.release()
            # Cleanup partial file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Failed to read frame {frame_idx}, stopping")
            break

        # Update progress
        app_state["batch_progress_current"] = frame_idx + 1
        progress_percent = int(((frame_idx + 1) / total_frames) * 100)
        app_state["video_cache"]["current_progress"] = progress_percent

        # Progress logging every 10 frames
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"Processing frame {frame_idx + 1}/{total_frames} ({progress_percent}%)")

        # Run inference (same logic as video_processing_loop)
        count = 0
        try:
            orig_h, orig_w = frame.shape[:2]

            # Resize for inference
            h, w = orig_h, orig_w
            scale = 1.0
            if max(h, w) > max_input_size:
                scale = max_input_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame

            image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Get model type
            model_type = getattr(model, 'model_type', 'text')

            # Prepare inputs
            inputs = None
            if prompt:
                # Text prompt mode
                if model_type != 'text':
                    print(f"[WARN] Batch frame {frame_idx}: Text prompts not supported with Sam3TrackerModel")
                else:
                    inputs = processor(
                        text=[prompt],
                        images=image_pil,
                        return_tensors="pt"
                    ).to(device)

            elif app_state.get("clicked_points"):
                # Interactive Segmentation: Use native point prompts
                if model_type != 'tracker':
                    print(f"[WARN] Batch frame {frame_idx}: Point prompts not supported with Sam3Model")
                else:
                    clicked_points = app_state.get("clicked_points", [])

                    # Convert to SAM 3 format: 4D nested list
                    input_points = [[]]
                    input_labels = [[]]

                    for point in clicked_points:
                        abs_x = int(point["x"] * orig_w)
                        abs_y = int(point["y"] * orig_h)
                        input_points[0].append([[abs_x, abs_y]])
                        input_labels[0].append([point["label"]])

                    inputs = processor(
                        images=image_pil,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt"
                    ).to(device)

            elif point_prompt:
                # Legacy single point prompt
                if model_type != 'tracker':
                    print(f"[WARN] Batch frame {frame_idx}: Point prompts not supported with Sam3Model")
                else:
                    abs_x = int(point_prompt["x"] * orig_w)
                    abs_y = int(point_prompt["y"] * orig_h)
                    label = point_prompt.get("label", 1)

                    inputs = processor(
                        images=image_pil,
                        input_points=[[[[abs_x, abs_y]]]],
                        input_labels=[[[label]]],
                        return_tensors="pt"
                    ).to(device)

            # Run inference
            if inputs:
                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-processing based on model type
                if model_type == 'text':
                    # Sam3Model: Use instance segmentation post-processing
                    results = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=confidence,
                        mask_threshold=mask_thresh,
                        target_sizes=[[orig_h, orig_w]]
                    )[0]

                else:  # tracker
                    # Sam3TrackerModel: Use masks post-processing
                    processed_masks = processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"],
                        mask_threshold=mask_thresh,
                        binarize=True
                    )

                    # Filter by IoU scores
                    iou_scores = outputs.iou_scores.squeeze(0)

                    final_masks_list = []
                    for i, (mask_batch, score_batch) in enumerate(zip(processed_masks[0], iou_scores)):
                        for mask, score in zip(mask_batch, score_batch):
                            if score >= confidence:
                                final_masks_list.append(mask.cpu().numpy())

                    results = {'masks': final_masks_list}

                # Extract masks
                final_masks = []
                if 'masks' in results and len(results['masks']) > 0:
                    for mask in results['masks']:
                        if isinstance(mask, torch.Tensor):
                            mask_np = mask.cpu().numpy()
                        else:
                            mask_np = np.array(mask)

                        if mask_np.dtype == bool:
                            mask_uint8 = (mask_np.astype(np.uint8)) * 255
                        elif mask_np.max() <= 1.0:
                            mask_uint8 = (mask_np * 255).astype(np.uint8)
                        else:
                            mask_uint8 = mask_np.astype(np.uint8)

                        final_masks.append(mask_uint8)

                # Note: With native point prompts, SAM 3 returns 1 mask per point
                # No post-filtering needed
                count = len(final_masks)

                # Draw masks on frame
                draw_masks(frame, final_masks)

        except Exception as e:
            print(f"[ERROR] Inference failed on frame {frame_idx}: {e}")
            torch.cuda.empty_cache()
            # Continue with unprocessed frame

        # Write processed frame to video file
        if out_writer:
            out_writer.write(frame)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        jpg_b64 = base64.b64encode(buffer).decode('utf-8')

        # Store in cache
        processed_frames.append({
            "idx": frame_idx,
            "base64": f"data:image/jpeg;base64,{jpg_b64}",
            "count": count,
        })

        # Yield control to allow other async tasks
        await asyncio.sleep(0.001)

    # 5. Finalize cache
    app_state["video_cache"]["frames"] = processed_frames
    app_state["video_cache"]["processing_complete"] = True
    app_state["batch_processing_active"] = False

    cap.release()
    if out_writer: out_writer.release()
    
    print(f"--- Batch Processing Complete: {len(processed_frames)} frames processed ---")
    print(f"--- Video saved to: {output_path} ---")
    return True

async def video_processing_loop(manager, app_state):
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0

    # Track active RTSP URL to detect changes
    current_active_rtsp = None

    # State Cache
    last_state_hash = None
    last_payload = None

    # Log Cache
    last_logged_count = -1

    # Preserve count for Image Mode (to prevent reset to 0 after caching)
    last_count = 0

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

        # Get performance parameters (for Video Mode optimization)
        processing_interval = app_state.get("processing_interval", 5)
        MAX_INPUT_SIZE = app_state.get("max_input_size", 1024)

        # RTSP State Management
        # If URL changed or became empty, release resource
        if rtsp_url != current_active_rtsp:
            if video_capture and input_mode == "rtsp": # Only affect RTSP capture
                 print(f"--- RTSP URL changed/cleared. Releasing capture. Old: {current_active_rtsp}, New: {rtsp_url} ---")
                 video_capture.release()
                 video_capture = None
                 app_state["video_capture"] = None  # Clear from app_state
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
                last_count = 0  # Reset count for Image Mode
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
                if video_capture:
                    video_capture.release()
                    video_capture = None
                    app_state["video_capture"] = None  # Clear from app_state
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
                    app_state["video_capture"] = None  # Clear from app_state
                    await asyncio.sleep(2)
                    continue
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                video_capture = None
                app_state["video_capture"] = None  # Clear from app_state
                continue
        elif input_mode == "video" and video_file_path and not rtsp_url and not uploaded_image_path:
            # VIDEO MODE: Check if batch processing or playback mode

            # If batch processing active, send progress updates
            if app_state.get("batch_processing_active", False):
                # Broadcast progress
                progress_current = app_state.get("batch_progress_current", 0)
                progress_total = app_state.get("batch_progress_total", 0)
                cache = app_state.get("video_cache", {})
                progress_percent = cache.get("current_progress", 0) if cache else 0

                # Create progress payload
                # Get current prompt for display
                current_prompt = app_state.get("prompt")
                point_prompt = app_state.get("point_prompt")

                progress_payload = {
                    "status": "batch_processing",
                    "video_frame": None,  # No frame during processing
                    "analytics": {
                        "input_mode": "video",
                        "process_status": "Processing...",
                        "detected_object": current_prompt if current_prompt else ("Selected Point" if point_prompt else "N/A"),
                        "batch_progress": {
                            "current": progress_current,
                            "total": progress_total,
                            "percent": progress_percent,
                        }
                    }
                }

                await manager.broadcast(json.dumps(progress_payload))
                await asyncio.sleep(0.1)
                continue

            # Check if cache exists and is complete
            cache = app_state.get("video_cache")
            if cache and cache.get("processing_complete", False):
                # PLAYBACK MODE: Read from cache

                # Handle seek request
                if video_seek_request is not None:
                    app_state["video_current_frame"] = video_seek_request
                    app_state["video_seek_request"] = None

                # Get current frame index
                current_idx = app_state.get("video_current_frame", 0)

                # Handle playback control
                if not video_playing:
                    # Paused: Send current frame repeatedly
                    if 0 <= current_idx < len(cache["frames"]):
                        cached_frame = cache["frames"][current_idx]

                        frame_metadata = {
                            "input_mode": "video",
                            "video_current_frame": current_idx,
                            "video_total_frames": cache["total_frames"],
                            "video_fps": cache["fps"],
                            "video_playing": False
                        }

                        count = cached_frame["count"]
                        status = "Approved" if count >= app_state["max_limit"] else "Waiting"
                        status_color = "green" if status == "Approved" else "orange"

                        analytics = {
                            "input_mode": "video",
                            "detected_object": cache["prompt"] or "N/A",
                            "count": count,
                            "max_limit": app_state["max_limit"],
                            "status": status,
                            "status_color": status_color,
                            "process_status": "Done",
                            "warning": None,
                            "trigger_sound": False
                        }
                        analytics.update(frame_metadata)

                        payload = {
                            "video_frame": cached_frame["base64"],
                            "analytics": analytics
                        }

                        await manager.broadcast(json.dumps(payload))
                        await asyncio.sleep(0.1)
                        continue

                # Playing: Advance frame
                # Loop video if at end
                if current_idx >= len(cache["frames"]):
                    current_idx = 0
                    app_state["video_current_frame"] = 0

                # Get cached frame
                cached_frame = cache["frames"][current_idx]

                # Advance to next frame
                app_state["video_current_frame"] = current_idx + 1

                # Prepare analytics
                frame_metadata = {
                    "input_mode": "video",
                    "video_current_frame": current_idx,
                    "video_total_frames": cache["total_frames"],
                    "video_fps": cache["fps"],
                    "video_playing": True
                }

                count = cached_frame["count"]
                status = "Approved" if count >= app_state["max_limit"] else "Waiting"
                status_color = "green" if status == "Approved" else "orange"

                analytics = {
                    "input_mode": "video",
                    "detected_object": cache["prompt"] or "N/A",
                    "count": count,
                    "max_limit": app_state["max_limit"],
                    "status": status,
                    "status_color": status_color,
                    "process_status": "Done",
                    "warning": None,
                    "trigger_sound": (status == "Approved" and app_state["sound_enabled"])
                }
                analytics.update(frame_metadata)

                payload = {
                    "video_frame": cached_frame["base64"],
                    "analytics": analytics
                }

                await manager.broadcast(json.dumps(payload))

                # Control playback speed
                fps = cache.get("fps", 30)
                if fps <= 0 or fps > 240:
                    fps = 30
                await asyncio.sleep(1.0 / fps)
                continue

            else:
                # No cache: Show RAW video playback (without masks) before batch processing
                # This allows user to preview/verify video before running segmentation

                # Open video for raw playback
                if not hasattr(app_state, '_preview_cap') or app_state.get('_preview_video_path') != video_file_path:
                    # Close old capture if exists
                    if hasattr(app_state, '_preview_cap') and app_state['_preview_cap'] is not None:
                        app_state['_preview_cap'].release()

                    # Open new video
                    app_state['_preview_cap'] = cv2.VideoCapture(video_file_path)
                    app_state['_preview_video_path'] = video_file_path
                    app_state['_preview_total_frames'] = int(app_state['_preview_cap'].get(cv2.CAP_PROP_FRAME_COUNT))
                    app_state['_preview_fps'] = app_state['_preview_cap'].get(cv2.CAP_PROP_FPS)

                    # Initialize playback state if not set
                    if "video_current_frame" not in app_state:
                        app_state["video_current_frame"] = 0
                    if "video_playing" not in app_state:
                        app_state["video_playing"] = False

                cap = app_state['_preview_cap']
                total_frames = app_state['_preview_total_frames']
                fps = app_state['_preview_fps']

                # Handle seek
                video_seek_request = app_state.get("video_seek_request")
                if video_seek_request is not None:
                    app_state["video_current_frame"] = video_seek_request
                    app_state["video_seek_request"] = None

                # Get current frame index
                current_idx = app_state.get("video_current_frame", 0)

                # Handle playback
                if video_playing:
                    # Playing: Advance frame
                    current_idx = (current_idx + 1) % total_frames
                    app_state["video_current_frame"] = current_idx

                # Read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                ret, frame = cap.read()

                if not ret:
                    # Reset to start if read fails
                    app_state["video_current_frame"] = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if ret:
                    # Encode raw frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    jpg_b64 = base64.b64encode(buffer).decode('utf-8')

                    # Get current prompt for display
                    current_prompt = app_state.get("prompt")
                    point_prompt = app_state.get("point_prompt")

                    preview_payload = {
                        "status": "waiting_batch",
                        "video_frame": f"data:image/jpeg;base64,{jpg_b64}",
                        "analytics": {
                            "input_mode": "video",
                            "process_status": "Ready",
                            "detected_object": current_prompt if current_prompt else ("Selected Point" if point_prompt else "N/A"),
                            "count": 0,
                            "video_current_frame": current_idx,
                            "video_total_frames": total_frames,
                            "video_fps": fps,
                            "video_playing": app_state.get("video_playing", False),
                        }
                    }
                    await manager.broadcast(json.dumps(preview_payload))

                # FPS-based sleep
                if fps <= 0 or fps > 240:
                    fps = 30
                await asyncio.sleep(1.0 / fps)
                continue
        else:
            await asyncio.sleep(0.5)
            continue

        # 2. Process Frame
        frame_counter += 1
        # For Image Mode: Preserve last count (prevents reset to 0 after caching)
        # For RTSP/Video: Always reset to 0 (continuous stream)
        count = last_count if input_mode == "image" else 0
        is_currently_processing = False  # Track if we're actively running inference THIS iteration

        # Check if we have any prompt (text, point, or clicked_points array)
        clicked_points = app_state.get("clicked_points", [])
        has_prompt = prompt or point_prompt or (clicked_points and len(clicked_points) > 0)

        should_process = (model and processor and has_prompt and should_segment)
        if input_mode in ["rtsp", "video"]:
             # Use dynamic processing_interval instead of hardcoded 5
             should_process = should_process and (frame_counter % processing_interval == 0)

        if should_process:
            is_currently_processing = True  # Set flag BEFORE inference starts

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
                # Get model type
                model_type = getattr(model, 'model_type', 'text')  # Default to 'text' if not set

                # Prepare Inputs
                if prompt:
                    # Text prompt mode (for Image/Video with text input)
                    if model_type != 'text':
                        print("[WARN] Text prompts not supported with Sam3TrackerModel. Use Sam3Model instead.")
                        # Skip inference
                    else:
                        inputs = processor(
                            text=[prompt],
                            images=image_pil,
                            return_tensors="pt"
                        ).to(device)

                elif app_state.get("clicked_points"):
                    # Interactive Segmentation: Use native point prompts
                    if model_type != 'tracker':
                        print("[WARN] Point prompts not supported with Sam3Model. Use Sam3TrackerModel instead.")
                        # Skip inference
                    else:
                        clicked_points = app_state.get("clicked_points", [])

                        # Convert to SAM 3 format: 4D nested list [batch][object][point][xy]
                        # Each object gets one point (the clicked coordinate)
                        input_points = [[]]  # Batch dimension
                        input_labels = [[]]  # Batch dimension

                        for point in clicked_points:
                            # Convert normalized coordinates to absolute pixels
                            abs_x = int(point["x"] * orig_w)
                            abs_y = int(point["y"] * orig_h)

                            # Add point: [[[x, y]]] format for each object
                            input_points[0].append([[abs_x, abs_y]])
                            input_labels[0].append([point["label"]])  # 1=positive, 0=negative

                        print(f"[DEBUG] Sending {len(input_points[0])} points to SAM 3 Tracker: {input_points}")

                        inputs = processor(
                            images=image_pil,
                            input_points=input_points,
                            input_labels=input_labels,
                            return_tensors="pt"
                        ).to(device)

                elif point_prompt:
                    # Legacy single point prompt support (backward compatibility)
                    if model_type != 'tracker':
                        print("[WARN] Point prompts not supported with Sam3Model. Use Sam3TrackerModel instead.")
                    else:
                        abs_x = int(point_prompt["x"] * orig_w)
                        abs_y = int(point_prompt["y"] * orig_h)
                        label = point_prompt.get("label", 1)

                        inputs = processor(
                            images=image_pil,
                            input_points=[[[[abs_x, abs_y]]]],  # 4D format
                            input_labels=[[[label]]],
                            return_tensors="pt"
                        ).to(device)

                # Run Inference
                if inputs:
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Post-processing based on model type
                    if model_type == 'text':
                        # Sam3Model: Use instance segmentation post-processing
                        results = processor.post_process_instance_segmentation(
                            outputs,
                            threshold=confidence,
                            mask_threshold=mask_thresh,
                            target_sizes=[[orig_h, orig_w]]
                        )[0]

                        raw_mask_count = len(results.get('masks', []))
                        print(f"[DEBUG] Sam3Model output: {raw_mask_count} masks")

                    else:  # tracker
                        # Sam3TrackerModel: Use masks post-processing
                        # Process masks
                        processed_masks = processor.post_process_masks(
                            outputs.pred_masks.cpu(),
                            inputs["original_sizes"],
                            mask_threshold=mask_thresh,
                            binarize=True
                        )

                        # Filter by IoU scores (confidence threshold)
                        iou_scores = outputs.iou_scores.squeeze(0)  # Remove batch dim

                        final_masks_list = []
                        for i, (mask_batch, score_batch) in enumerate(zip(processed_masks[0], iou_scores)):
                            # mask_batch shape: (num_masks, H, W)
                            # score_batch shape: (num_masks,)
                            for mask, score in zip(mask_batch, score_batch):
                                if score >= confidence:
                                    final_masks_list.append(mask.cpu().numpy())

                        # Create results dict compatible with existing code
                        results = {'masks': final_masks_list}

                        print(f"[DEBUG] Sam3TrackerModel output: {len(final_masks_list)} masks (iou >= {confidence})")

                    # Note: With native point prompts, SAM 3 Tracker returns 1 mask per point
                    # Masks are already matched to clicked points

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

                    # Update last_count for Image Mode persistence
                    if input_mode == "image":
                        last_count = count

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

        # Determine Process Status for UI (✅ FIXED: Ready → Processing → Done flow)
        if input_mode == "image":
            # IMAGE MODE: Set to Done immediately after inference completes
            if should_segment and count > 0:
                # Segmentation finished with results
                process_status = "Done"
            elif should_segment and count == 0:
                # Segmentation finished but no results (filter too strict or no objects)
                process_status = "Done"
            elif should_segment and is_currently_processing:
                # Currently processing (rare, might only show briefly)
                process_status = "Processing..."
            else:
                # No segmentation active
                process_status = "Ready"
        else:
            # RTSP/VIDEO MODE: Keep existing logic for continuous streams
            if is_currently_processing:
                # Inference is actively running THIS frame
                process_status = "Processing..."
            elif should_segment and count > 0:
                # Segmentation is enabled AND we have results (from previous processed frame)
                process_status = "Done"
            elif should_segment and count == 0:
                # Segmentation enabled but no results yet (or empty result)
                process_status = "Processing..." if should_process else "Done"
            else:
                # Segmentation not enabled (Clear Mask or no prompt)
                process_status = "Ready"

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