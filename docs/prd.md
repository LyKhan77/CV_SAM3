# Product Requirements Document (PRD): AI Computer Vision Monitoring Dashboard

| Attribute | Details |
| :--- | :--- |
| **Project Name** | AI CV Monitoring Dashboard |
| **Model** | META SAM 3 (Segment Anything Model 3) |
| **Version** | 1.0 |
| **Date** | November 25, 2024 |

---

## 1. Overview

The system is an AI-powered monitoring dashboard that ingests real-time video feeds (RTSP), applies the SAM 3 (Segment Anything Model) from Meta to detect and segment specific objects based on user text prompts, and tracks the quantity of those objects against a user-defined target limit.

## 2. Functional Requirements

### 2.1. AI Engine (Backend)
- **Model Implementation**: The backend must host and run Meta SAM 3.
- **Prompting Mechanism**: The model must accept text-based prompts (e.g., "Cardboard box", "Helmet") to define the segmentation mask.
- **Inference**:
    - Process video frames to identify instances of the prompted object.
    - Return the count of unique instances detected in the current frame or time window.

### 2.2. Streaming Panel
- **RTSP Ingestion**:
    - **Input**: The backend must expose an endpoint to accept an RTSP URL string (e.g., `rtsp://192.168.x.x...`).
    - **Processing**: Upon receiving the URL (triggered by "ENTER"), the system must initiate the connection to the camera stream.
    - **Output**: The backend must transcode or stream the video feed (with or without segmentation overlays) back to the frontend container.
- **Latency**: Minimal latency is required for real-time monitoring.

### 2.3. Prompt & Description
- **Object Detection Input**:
    - **Input**: A text field accepting the target object name.
    - **Action**: Pressing "ENTER" sends the text to the backend to re-configure the SAM 3 inference target.
- **Description Feedback**:
    - **Output**: The backend must acknowledge the prompt and return the active object description to be displayed in the Description panel (e.g., "Tracking: Red Helmet").

## 3. Business Logic & State Management (Summary Panel)
This section defines how the backend calculates status and triggers events based on the detected count versus the user-defined limit.

### 3.1. Maximum Limit (`max_limit`)
- **Input**: Accepts a numeric integer value.
- **Storage**: The backend stores this value as the target threshold.

### 3.2. Total Detected (`count`)
- **Calculation**: The AI engine counts the number of segmented objects matching the prompt.
- **Data Payload**: The backend must send a JSON payload containing:
    - `current_count`: The number of detected objects.
    - `max_limit`: The set limit.
    - `progress`: A calculated percentage or ratio for the progress bar (`count / max_limit`).

### 3.3. Status Logic
The status logic implies a "Target/Quota" system (detecting enough items).
- **Logic**:
    - **IF `count >= max_limit`**:
        - **Status**: "Approved"
        - **Color Indicator**: Green
    - **IF `count < max_limit`**:
        - **Status**: "Waiting"
        - **Color Indicator**: Orange

### 3.4. Notification System
- **Activation**: Controlled by a boolean flag ("Activate Notification Sound" checkbox).
- **Trigger Condition**:
    - If `sound_enabled == TRUE` AND (`Status == Approved` OR `count >= max_limit`).
- **Action**: The backend sends a trigger signal (e.g., `alert: true`) to the frontend to play the sound.

## 4. API / Data Flow Specifications

### 4.1. WebSocket Endpoint (`/ws/monitor`)
To ensure real-time updates, a WebSocket connection is recommended over HTTP polling.

**Server-to-Client Payload** (emitted every frame or detection interval):
```json
{
  "timestamp": 1716634200,
  "video_frame": "<base64_encoded_image_or_stream_url>",
  "analytics": {
    "detected_object": "Box",
    "count": 50,
    "max_limit": 50,
    "status": "Approved",
    "status_color": "success", // or "green"
    "trigger_sound": true
  }
}
```

### 4.2. Control Endpoints (HTTP POST)
- **Set Stream**: `POST /api/config/stream`
  - **Body**: `{ "url": "rtsp://..." }`
- **Set Prompt**: `POST /api/config/prompt`
  - **Body**: `{ "object_name": "Forklift" }`
- **Set Limit**: `POST /api/config/limit`
  - **Body**: `{ "value": 100 }`
- **Toggle Sound**: `POST /api/config/sound`
  - **Body**: `{ "enabled": true }`

## 5. Non-Functional Requirements
- **Performance**: Inference must run at a frame rate sufficient for monitoring (e.g., min 5 FPS) depending on GPU availability.
- **Scalability**: The backend structure should support handling the RTSP stream and SAM 3 inference asynchronously to prevent blocking the API.
