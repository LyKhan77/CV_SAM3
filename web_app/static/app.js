// --- 0. Constants and Configuration ---
const API_BASE_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/monitor";

// --- 1. Utilities (Time) ---
function updateTime() {
    const now = new Date();
    document.getElementById('clock-time').textContent = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    document.getElementById('clock-date').textContent = now.toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'short', year: 'numeric' });
}
setInterval(updateTime, 1000);
updateTime();

// --- 2. State Management ---
let frontendState = {
    currentCount: 0,
    isClickSelectMode: false,
};

// --- 3. WebSocket Connection ---
const socket = new WebSocket(WS_URL);
socket.onopen = () => console.log("WebSocket connection established.");
socket.onclose = () => console.log("WebSocket connection closed.");
socket.onerror = (error) => console.error("WebSocket error:", error);
socket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    } catch (e) {
        console.error("Failed to parse WebSocket message:", e, "Raw data:", event.data);
    }
};

// --- 4. API Call Functions ---
async function postData(endpoint, body) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return await response.json();
    } catch (e) {
        console.error(`Failed to post to ${endpoint}:`, e);
        alert(`Error communicating with the backend. Is it running?`);
    }
}

// --- 5. UI Event Handlers ---
function activateStream() {
    const url = document.getElementById('rtsp-input').value;
    if (!url.trim()) return alert("Please enter a valid RTSP URL");
    
    document.getElementById('stream-placeholder').innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to RTSP...</p>';
    
    postData("/api/config/stream", { url }).then(data => {
        if(data) {
            console.log("Stream activation response:", data);
            document.getElementById('stream-placeholder').classList.add('hidden');
            document.getElementById('live-indicator').classList.remove('hidden');
            document.getElementById('mock-video-feed').classList.remove('hidden');
        }
    });
}

function processPrompt() {
    const inputVal = document.getElementById('prompt-input').value;
    if (!inputVal.trim()) return;
    document.getElementById('object-description').innerHTML = '<div class="loader"></div>';
    postData("/api/config/prompt", { object_name: inputVal }).then(data => console.log("Prompt set response:", data));
}

function updateLimit() {
    const limit = parseInt(document.getElementById('max-limit').value) || 100;
    postData("/api/config/limit", { value: limit }).then(data => console.log("Limit set response:", data));
}

function updateSoundSetting() {
    const enabled = document.getElementById('sound-toggle').checked;
    postData("/api/config/sound", { enabled }).then(data => console.log("Sound setting response:", data));
}

function updateModelConfig() {
    const confidence = parseInt(document.getElementById('confidence-slider').value) / 100;
    const displayMode = document.getElementById('display-mode-toggle').checked ? "bounding_box" : "segmentation";
    postData("/api/config/model", { confidence, display_mode: displayMode }).then(data => console.log("Model config response:", data));
}

function handleVideoClick(event) {
    if (!frontendState.isClickSelectMode) return;

    const videoFeed = document.getElementById('mock-video-feed');
    const rect = videoFeed.getBoundingClientRect();

    // Calculate click coordinates relative to the image's display size
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Normalize coordinates to the image's natural dimensions
    const normalX = x / rect.width;
    const normalY = y / rect.height;

    // The backend expects coordinates based on the original video frame size.
    // We send normalized coordinates, and the backend will scale them.
    const point_prompt = {
        type: "point_prompt",
        points: { x: normalX, y: normalY }
    };

    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(point_prompt));
        document.getElementById('object-description').innerHTML = '<div class="loader"></div>';
        // Disable click-select mode after one click
        document.getElementById('select-object-toggle').checked = false;
        toggleClickSelectMode(); 
    } else {
        console.error("WebSocket is not open. Cannot send point prompt.");
    }
}

function toggleClickSelectMode() {
    frontendState.isClickSelectMode = document.getElementById('select-object-toggle').checked;
    const videoFeed = document.getElementById('mock-video-feed');
    const promptInput = document.getElementById('prompt-input');
    
    if (frontendState.isClickSelectMode) {
        videoFeed.classList.add('cursor-crosshair');
        promptInput.disabled = true;
        promptInput.placeholder = "Click on video to select object";
        promptInput.value = "";
        // Clear text prompt in backend
        postData("/api/config/prompt", { object_name: "" });
    } else {
        videoFeed.classList.remove('cursor-crosshair');
        promptInput.disabled = false;
        promptInput.placeholder = "{Object Name}";
    }
}

// --- 6. DOM Element Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Existing listeners
    document.getElementById('max-limit').addEventListener('change', updateLimit);
    document.getElementById('sound-toggle').addEventListener('change', updateSoundSetting);
    // New listeners
    const confidenceSlider = document.getElementById('confidence-slider');
    confidenceSlider.addEventListener('input', () => {
        document.getElementById('confidence-value').textContent = `${confidenceSlider.value}%`;
    });
    confidenceSlider.addEventListener('change', updateModelConfig);
    
    const displayModeToggle = document.getElementById('display-mode-toggle');
    displayModeToggle.addEventListener('change', () => {
        document.getElementById('display-mode-label').textContent = displayModeToggle.checked ? "Bounding Box" : "Segmentation";
        updateModelConfig();
    });

    document.getElementById('select-object-toggle').addEventListener('change', toggleClickSelectMode);
    document.getElementById('mock-video-feed').addEventListener('click', handleVideoClick);
});

// --- 7. Dashboard Update Logic ---
function updateDashboard(data) {
    const analytics = data.analytics;
    // ... (rest of the function is the same as before, no changes needed here)
    const countEl = document.getElementById('detected-count');
    const descEl = document.getElementById('object-description');
    const statusBadge = document.getElementById('status-badge');
    const progressBar = document.getElementById('progress-bar');
    const progressLegend = document.getElementById('progress-legend');
    const videoFeed = document.getElementById('mock-video-feed');
    
    animateValue(countEl, frontendState.currentCount, analytics.count, 500);
    frontendState.currentCount = analytics.count;

    if(data.video_frame && data.video_frame.startsWith('data:image')) {
       videoFeed.src = data.video_frame;
    }

    if (analytics.detected_object && analytics.detected_object !== "N/A") {
        descEl.innerHTML = `Detecting: <span class="font-bold text-primary">${analytics.detected_object}</span>`;
    } else if (analytics.detected_object === null) {
        descEl.textContent = "{Object Description}";
    }
    
    let percentage = (analytics.count / analytics.max_limit) * 100;
    if (percentage > 100) percentage = 100;
    progressBar.style.width = `${percentage}%`;

    progressLegend.textContent = `${analytics.count}/${analytics.max_limit}`;
    document.getElementById('max-limit').value = analytics.max_limit;

    statusBadge.textContent = analytics.status;
    const statusColor = analytics.status_color.toLowerCase();
    if (statusColor === 'success' || statusColor === 'green' || statusColor === 'approved') {
        statusBadge.className = "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-green-200 text-green-800 transition-colors duration-300";
        progressBar.classList.add('bg-primary');
        progressBar.classList.remove('bg-red-600');
    } else if (statusColor === 'orange' || statusColor === 'waiting') {
        statusBadge.className = "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-orange-200 text-orange-800 transition-colors duration-300";
        progressBar.classList.add('bg-primary');
        progressBar.classList.remove('bg-red-600');
    } else {
        statusBadge.className = "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-red-200 text-red-800 transition-colors duration-300 animate-pulse";
        progressBar.classList.remove('bg-primary');
        progressBar.classList.add('bg-red-600');
    }

    if (analytics.trigger_sound) triggerNotification();
}

function triggerNotification() {
    const audio = document.getElementById('notification-sound');
    audio.currentTime = 0;
    audio.play().catch(e => console.log("Audio play failed:", e));
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}

// --- 8. Image Upload Functions ---
function setupImageUpload() {
    const uploadArea = document.getElementById('image-upload-area');
    const fileInput = document.getElementById('image-input');
    const previewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('uploaded-image-preview');
    const clearBtn = document.getElementById('clear-image-btn');

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    clearBtn.addEventListener('click', clearUploadedImage);

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-primary', 'bg-primary/10');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-primary', 'bg-primary/10');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-primary/10');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type and size
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    document.getElementById('image-upload-area').innerHTML = `
        <i class="fa-solid fa-spinner fa-spin text-gray-400 text-4xl mb-3"></i>
        <p class="text-gray-600 font-medium">Uploading...</p>
    `;

    fetch(`${API_BASE_URL}/api/upload/image`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Restore upload area
        document.getElementById('image-upload-area').innerHTML = `
            <input type="file" id="image-input" accept=".jpeg,.jpg,.png" class="hidden">
            <i class="fa-solid fa-cloud-upload-alt text-gray-400 text-4xl mb-3"></i>
            <p class="text-gray-600 font-medium">Click to upload image</p>
            <p class="text-gray-400 text-sm">or drag and drop</p>
            <p class="text-gray-400 text-xs mt-2">JPEG, JPG, PNG up to 10MB</p>
        `;

        // Re-attach event listener
        document.getElementById('image-input').addEventListener('change', handleFileSelect);
        document.getElementById('image-upload-area').addEventListener('click', () => {
            document.getElementById('image-input').click();
        });

        if (data.status === 'success') {
            displayUploadedImage(file);
            console.log('Image uploaded successfully:', data.message);
        } else {
            alert('Upload failed: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        alert('Upload failed');

        // Restore upload area on error
        document.getElementById('image-upload-area').innerHTML = `
            <input type="file" id="image-input" accept=".jpeg,.jpg,.png" class="hidden">
            <i class="fa-solid fa-cloud-upload-alt text-gray-400 text-4xl mb-3"></i>
            <p class="text-gray-600 font-medium">Click to upload image</p>
            <p class="text-gray-400 text-sm">or drag and drop</p>
            <p class="text-gray-400 text-xs mt-2">JPEG, JPG, PNG up to 10MB</p>
        `;
    });
}

function displayUploadedImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('uploaded-image-preview');
        const container = document.getElementById('image-preview-container');
        preview.src = e.target.result;
        container.classList.remove('hidden');

        // Update video feed to show uploaded image
        document.getElementById('mock-video-feed').src = e.target.result;
        document.getElementById('mock-video-feed').classList.remove('hidden');
        document.getElementById('stream-placeholder').classList.add('hidden');
        document.getElementById('live-indicator').classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function clearUploadedImage() {
    postData("/api/config/clear-image", {}).then(data => {
        if (data.status === 'success') {
            document.getElementById('image-preview-container').classList.add('hidden');
            document.getElementById('image-input').value = '';
            console.log('Image cleared');
        }
    });
}

// --- 9. Smart Prompt Dropdown Functions ---
function setupSmartPromptDropdown() {
    const promptInput = document.getElementById('prompt-input');
    const dropdownToggle = document.getElementById('dropdown-toggle');
    const suggestionsPanel = document.getElementById('object-suggestions');

    // Toggle dropdown
    dropdownToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHidden = suggestionsPanel.classList.contains('hidden');
        if (isHidden) {
            suggestionsPanel.classList.remove('hidden');
            // Position dropdown correctly
            const inputRect = promptInput.getBoundingClientRect();
            const parentRect = promptInput.parentElement.getBoundingClientRect();

            suggestionsPanel.style.width = '100%';
            suggestionsPanel.style.left = '0';
            suggestionsPanel.style.top = `${inputRect.height + 4}px`;
            suggestionsPanel.style.maxWidth = 'none';
        } else {
            suggestionsPanel.classList.add('hidden');
        }
    });

    // Handle suggestion clicks
    document.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', (e) => {
            promptInput.value = e.target.textContent.trim();
            suggestionsPanel.classList.add('hidden');
            promptInput.focus();
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', () => {
        suggestionsPanel.classList.add('hidden');
    });

    // Prevent dropdown close when clicking inside
    suggestionsPanel.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Close dropdown on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            suggestionsPanel.classList.add('hidden');
        }
    });
}

// Initialize image upload and smart dropdown on DOM load
document.addEventListener('DOMContentLoaded', () => {
    setupImageUpload();
    setupSmartPromptDropdown();
});