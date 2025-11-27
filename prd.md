This is a comprehensive and detailed Product Requirement Document (PRD). It expands on the technical specifics, UI component definitions (with specific Tailwind classes), and data flow logic to ensure zero ambiguity during the development phase.

-----

# Product Requirement Document (PRD): AI Vision Monitor

| **Project Name** | AI Vision Monitor Dashboard |
| :--- | :--- |
| **Version** | 2.0 (Detailed Specification) |
| **Date** | November 26, 2025 |
| **Tech Stack** | Python Flask, HTML5, Tailwind CSS, Vanilla JS |
| **AI Engine** | Meta SAM 3 (Segment Anything Model) |
| **Theme Color** | `#003473` (Deep Navy) |

-----

## 1\. Executive Summary

The **AI Vision Monitor** is a web-based compliance tool. It uses Computer Vision (SAM 3) to count specific objects in an uploaded image based on a natural language prompt. The system compares the detected count against a user-defined threshold to determine if a specific operational condition is "Approved" or "Waiting."

-----

## 2\. User Experience (UX) & Interface Specifications

**Framework:** HTML5 + Tailwind CSS (via CDN for development).
**Primary Color:** `#003473` (Used for Buttons, Headers, Progress Bars).

### 2.1 Layout Structure

  * **Global Container:** `min-h-screen bg-gray-50 flex flex-col`.
  * **Header:** Simple Navbar with Logo/Title using `bg-[#003473] text-white`.
  * **Main Grid:** A 2-column layout on large screens (`lg:grid lg:grid-cols-12 gap-6 p-6`).
      * **Left Column (Input/View):** `col-span-8`
      * **Right Column (Controls/Stats):** `col-span-4`

### 2.2 Component Details

#### A. Main Container (Image Input & View)

  * **Initial State (Upload Mode):**
      * Area with dashed border: `border-2 border-dashed border-gray-300 rounded-xl`.
      * Center content: Icon + "Click or Drag to Upload".
      * Hidden Input: `<input type="file" id="imageInput" accept=".jpg, .jpeg, .png">`.
  * **Preview State:**
      * Displays the raw image immediately after selection.
      * Class: `w-full h-auto rounded-lg shadow-sm object-contain`.
  * **Result State:**
      * Replaces the raw image with the **Processed Image** (containing SAM 3 segmentation masks/bounding boxes) returned from the server.

#### B. Prompt & Control Panel (Right Sidebar)

  * **Input Field (Object Name):**
      * Label: "Object to Detect".
      * Style: `w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-[#003473]`.
      * Placeholder: "e.g., helmet, box, person".
  * **Run Button:**
      * Style: `w-full bg-[#003473] hover:bg-blue-900 text-white font-bold py-3 rounded-lg transition duration-200`.
      * **Loading State:** Changes text to "Processing..." and disables button during API call.

#### C. Summary Dashboard (Right Sidebar - Below Prompt)

This section uses a card-based layout (`bg-white p-4 rounded-xl shadow-sm`).

1.  **"Maximum Limit" Card**

      * Input: `<input type="number">`.
      * Visual: Large, prominent input field.
      * Purpose: Sets the denominator for the logic check.

2.  **"Total Detected" Card**

      * **Counter:** Large typography displaying `{count}`.
      * **Progress Bar:**
          * Track: `w-full h-4 bg-gray-200 rounded-full`.
          * Fill: `h-4 rounded-full bg-[#003473] transition-all duration-500`.
      * **Label:** Small text `{count} / {max_limit}`.

3.  **"Status" Card**

      * **Container:** `flex items-center justify-center p-4 rounded-lg border-2`.
      * **Dynamic States (Managed by JS):**
          * **Waiting (Default/Fail):**
              * UI: `bg-orange-50 border-orange-200 text-orange-700`.
              * Icon/Text: "⏳ Waiting" or "⚠️ Below Limit".
          * **Approved (Success):**
              * UI: `bg-green-50 border-green-200 text-green-700`.
              * Icon/Text: "✅ Approved".

-----

## 3\. Technical Architecture

### 3.1 File & Folder Structure

Strict separation ensures the Flask `render_template` engine finds HTML, and the browser finds CSS/JS/Images.

```text
/ai_vision_dashboard
│
├── app.py                     # Main Flask Application Entry Point
├── requirements.txt           # Dependencies (flask, torch, numpy, cv2, segment-anything)
├── .env                       # Environment variables (optional)
│
├── model/                     # AI Engine logic
│   ├── __init__.py
│   ├── sam_engine.py          # Class: SamPredictorWrapper
│   └── weights/               # Local storage for model checkpoints
│       └── sam_vit_h_4b8939.pth
│
├── static/                    # Public assets served by Flask
│   ├── css/
│   │   └── output.css         # (Optional) If compiling custom Tailwind
│   ├── js/
│   │   └── dashboard.js       # Main DOM manipulation logic
│   └── uploads/               # Temp folder for processed images
│       └── .gitkeep
│
└── templates/                 # HTML Templates
    └── index.html             # The Dashboard UI
```

### 3.2 Data Flow

1.  **User Action:** User selects image -\> inputs text "Worker" -\> Inputs Limit "5" -\> Clicks "Run".
2.  **Frontend (JS):** Creates `FormData` object containing the file and text. Sends `POST` to `/analyze`.
3.  **Backend (Flask):**
      * Saves raw image to `static/uploads/temp_raw.jpg`.
      * Passes image path + prompt to `model/sam_engine.py`.
4.  **AI Engine (SAM 3):**
      * Converts prompt to embedding.
      * Runs inference on the image.
      * Generates masks.
      * Draws contours on the image using OpenCV.
      * Saves result to `static/uploads/temp_result.jpg`.
      * Returns: `count` (integer).
5.  **Response:** Flask sends JSON: `{ "count": 3, "image_url": "/static/uploads/temp_result.jpg?t=timestamp" }`.
6.  **Frontend (JS):**
      * Updates `img src`.
      * Updates Count text.
      * Runs Logic: `if (count >= limit) { setStatus('Approved') } else { setStatus('Waiting') }`.

-----

## 4\. API Specification

### Endpoint: Process Image

  * **URL:** `/analyze`
  * **Method:** `POST`
  * **Content-Type:** `multipart/form-data`

**Request Parameters:**
| Key | Type | Description |
| :--- | :--- | :--- |
| `file` | Binary (File) | The uploaded image (.jpg, .png). |
| `prompt` | String | The text prompt for SAM 3. |

**Success Response (200 OK):**

```json
{
  "success": true,
  "message": "Analysis complete",
  "data": {
    "detected_count": 5,
    "prompt_used": "helmet",
    "result_image_url": "/static/uploads/processed_12345.jpg"
  }
}
```

**Error Response (400/500):**

```json
{
  "success": false,
  "error": "Model failed to load or invalid image format."
}
```

-----

## 5\. Logic & Algorithms

### 5.1 Status Logic (JavaScript)

The frontend is responsible for the final status determination to reduce server load for simple math.

```javascript
// Pseudo-code for Status Card
const maxLimit = parseInt(document.getElementById('maxLimitInput').value) || 0;
const detectedCount = apiResponse.data.detected_count;

// Progress Bar Calculation
const percentage = maxLimit > 0 ? (detectedCount / maxLimit) * 100 : 0;
progressBar.style.width = `${Math.min(percentage, 100)}%`;

// Status Determination
if (detectedCount >= maxLimit && maxLimit > 0) {
    // Apply Approved Theme
    statusCard.className = "bg-green-50 border-green-500 text-green-700 ...";
    statusText.innerText = "Approved";
} else {
    // Apply Waiting Theme
    statusCard.className = "bg-orange-50 border-orange-500 text-orange-700 ...";
    statusText.innerText = "Waiting";
}
```

### 5.2 AI Inference Logic (Python)

  * **Input:** Image Array, Text Prompt.
  * **Process:**
    1.  Use GroundingDINO (or similar simplified text-to-box method often paired with SAM) or SAM's native text prompt capabilities if available in the specific v3 implementation used.
    2.  *Fallback approach if specific text-prompt SAM 3 is complex:* Use a lightweight detector (like YOLO) to get bounding boxes for the prompt, then feed boxes to SAM for precise segmentation.
    3.  **Counting:** Count the number of unique boolean masks generated.
    4.  **Visualization:** Overlay semi-transparent masks (color: `#003473` with alpha 0.4) on the image.

-----

## 6\. Requirements for Development Environment

  * **OS:** Linux (Ubuntu) preferred (matches user persona).
  * **GPU:** NVIDIA GPU with CUDA support (Recommended for SAM 3).
  * **Python:** 3.9 or 3.10.
  * **Key Libraries:**
      * `flask`
      * `torch`, `torchvision`
      * `opencv-python`
      * `segment-anything` (Meta's repo)
      * `transformers` (if using HuggingFace implementation)