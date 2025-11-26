import asyncio
import cv2
import base64
import json
import torch
import numpy as np
from PIL import Image
# Using SAM-3 specific classes for proper functionality
from transformers import Sam3Processor, Sam3Model

def draw_masks(frame, masks, display_mode):
    """
    Draw masks on frame based on display mode (segmentation or bounding box)
    Fixed to handle tensor to numpy conversion properly
    """
    if display_mode == "bounding_box":
        # Draw bounding boxes
        for mask in masks:
            # Convert tensor to numpy properly
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy().astype(np.uint8)
            else:
                mask_np = mask.astype(np.uint8)

            x, y, w, h = cv2.boundingRect(mask_np)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Draw segmentation contours (default behavior)
        for mask in masks:
            # Convert tensor to numpy properly
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = mask

            # Ensure binary format for contour detection
            binary_mask = (mask_np > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)


def process_sam3_outputs(outputs, inputs, processor, confidence_threshold):
    """
    Unified function to process SAM-3 outputs for both text and point prompts
    """
    try:
        # Extract masks using SAM-3 post-processing
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )[0]

        # Convert to binary masks
        binary_masks = (masks > 0.5).squeeze(1)

        # Extract scores if available, otherwise use default confidence
        if hasattr(outputs, 'iou_scores'):
            scores = outputs.iou_scores.squeeze(1).cpu().numpy()
        else:
            scores = [1.0] * len(binary_masks)

        # Filter by confidence threshold
        filtered_masks = []
        for i, mask in enumerate(binary_masks):
            if i < len(scores) and scores[i] >= confidence_threshold:
                filtered_masks.append(mask)

        return filtered_masks

    except Exception as e:
        print(f"[ERROR] Post-processing failed: {e}")
        return []


def load_model():
    """
    Loads the SAM-3 model and processor using SAM-3 specific classes for proper functionality.
    """
    try:
        print("--- Loading SAM-3 Model and Processor using Sam3Model/Sam3Processor ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model_id = "facebook/sam3"

        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id).to(device)

        print("--- Model and Processor Loaded Successfully ---")
        return model, processor
    except Exception as e:
        print(f"FATAL: Failed to load AI model: {e}")
        return None, None

async def video_processing_loop(manager, app_state):
    """
    Modified to handle both RTSP streams and static images.
    """
    print("--- Video Processing Loop Started ---")
    video_capture = None
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame for static images

    while True:
        rtsp_url = app_state.get("rtsp_url")
        uploaded_image_path = app_state.get("uploaded_image_path")
        model = app_state.get("model")
        processor = app_state.get("processor")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        frame = None

        # Handle static image
        if uploaded_image_path and not rtsp_url:
            if video_capture:
                video_capture.release()
                video_capture = None

            frame = cv2.imread(uploaded_image_path)
            if frame is None:
                print(f"ERROR: Could not read uploaded image: {uploaded_image_path}")
                await asyncio.sleep(1)
                continue

        # Handle RTSP stream
        elif rtsp_url and not uploaded_image_path:
            if video_capture is None:
                try:
                    print(f"Attempting to connect to RTSP stream: {rtsp_url}")
                    video_capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if not video_capture.isOpened(): raise ValueError("Could not open stream.")
                    print("--- RTSP Stream Connected ---")
                except Exception as e:
                    print(f"ERROR: Failed to connect: {e}")
                    video_capture = None
                    app_state["rtsp_url"] = None
                    await asyncio.sleep(5)
                    continue

            ret, frame = video_capture.read()
            if not ret:
                print("--- Could not read frame, disconnecting. ---")
                video_capture.release()
                video_capture = None
                app_state["rtsp_url"] = None
                continue

        else:
            # No source available
            if video_capture:
                video_capture.release()
                video_capture = None
            await asyncio.sleep(1)
            continue

        frame_counter += 1
        count = 0
        prompt = app_state.get("prompt")
        point_prompt = app_state.get("point_prompt")
        confidence_threshold = app_state.get("confidence_threshold", 0.5)
        display_mode = app_state.get("display_mode", "segmentation")

        # Enhanced debugging logging
        print(f"[DEBUG] Processing frame {frame_counter}")
        print(f"[DEBUG] Text prompt: {prompt}")
        print(f"[DEBUG] Point prompt: {point_prompt}")
        print(f"[DEBUG] Confidence threshold: {confidence_threshold}")
        print(f"[DEBUG] Display mode: {display_mode}")
        print(f"[DEBUG] Image source: {'RTSP' if rtsp_url else 'Local' if uploaded_image_path else 'None'}")

        # Early exit if no prompts to reduce lag
        if not prompt and not point_prompt:
            # Skip AI processing if no prompts active
            pass

        # Process text prompts
        if model and processor and prompt and (frame_counter % PROCESS_EVERY_N_FRAMES == 0):
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                print(f"[DEBUG] Processing text prompt: {prompt}")

                # SAM-3 text prompt processing
                inputs = processor(
                    text=[prompt],
                    images=image_pil,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Use unified post-processing function
                filtered_masks = process_sam3_outputs(outputs, inputs, processor, confidence_threshold)

                count = len(filtered_masks)
                print(f"[DEBUG] Text prompt generated {count} masks after filtering")

                # Apply conditional rendering
                draw_masks(frame, filtered_masks, display_mode)

            except Exception as e:
                print(f"[ERROR] Text prompt inference failed: {e}")
                # Continue processing without AI overlay

        # Process point prompts
        elif model and processor and point_prompt and (frame_counter % PROCESS_EVERY_N_FRAMES == 0):
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                print(f"[DEBUG] Processing point prompt: {point_prompt}")

                # Convert normalized coordinates to absolute
                h, w = image_pil.size[::-1]
                abs_x = int(point_prompt["x"] * w)
                abs_y = int(point_prompt["y"] * h)

                # Try different SAM-3 point prompt formats
                inputs = None
                formats_to_try = [
                    # Approach 1: 2D array format
                    lambda: processor(
                        images=image_pil,
                        input_points=[[abs_x, abs_y]],
                        input_labels=[[1]],
                        return_tensors="pt"
                    ),
                    # Approach 2: 2D array without labels
                    lambda: processor(
                        images=image_pil,
                        input_points=[[abs_x, abs_y]],
                        return_tensors="pt"
                    ),
                    # Approach 3: Single point format
                    lambda: processor(
                        images=image_pil,
                        input_points=[abs_x, abs_y],
                        return_tensors="pt"
                    )
                ]

                for i, format_func in enumerate(formats_to_try):
                    try:
                        inputs = format_func().to(device)
                        print(f"[DEBUG] Point prompt format {i+1} succeeded")
                        break
                    except Exception as format_error:
                        print(f"[DEBUG] Point prompt format {i+1} failed: {format_error}")
                        if i == len(formats_to_try) - 1:
                            raise format_error

                if inputs is None:
                    raise ValueError("Failed to create inputs with any point format")

                with torch.no_grad():
                    outputs = model(**inputs)

                # Use unified post-processing function
                filtered_masks = process_sam3_outputs(outputs, inputs, processor, confidence_threshold)

                count = len(filtered_masks)
                print(f"[DEBUG] Point prompt generated {count} masks after filtering")

                # Apply conditional rendering
                draw_masks(frame, filtered_masks, display_mode)

            except Exception as e:
                print(f"[ERROR] Point prompt inference failed: {e}")
                # Continue processing without AI overlay

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
