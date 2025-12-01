// --- 0. Constants and Configuration ---
const API_BASE_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/monitor";

// --- 1. Utilities (Time) ---
function updateTime() {
    const timeEl = document.getElementById('clock-time');
    const dateEl = document.getElementById('clock-date');

    if (!timeEl || !dateEl) {
        console.warn('Clock elements not found');
        return;
    }

    const now = new Date();
    timeEl.textContent = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    dateEl.textContent = now.toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'short', year: 'numeric' });
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

// Prevent WebSocket errors when closed
socket.onclose = () => console.log("WebSocket connection closed.");

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
        // Only alert for critical config failures, not routine checks
        if (endpoint.includes('config/stream')) alert(`Error communicating with the backend. Is it running?`);
    }
}

// --- 5. UI Event Handlers ---
function activateStream() {
    const urlEl = document.getElementById('rtsp-url-input');
    const placeholderEl = document.getElementById('stream-placeholder');
    const liveIndicatorEl = document.getElementById('live-indicator');
    const videoFeedEl = document.getElementById('mock-video-feed');

    if (!urlEl || !urlEl.value.trim()) {
        alert("Please enter a valid RTSP URL");
        return;
    }

    placeholderEl.innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to RTSP...</p>';

    postData("/api/config/stream", { url: urlEl.value }).then(data => {
        if(data && data.status === 'success') {
            console.log("Stream activation response:", data);
            placeholderEl.classList.add('hidden');
            if (liveIndicatorEl) liveIndicatorEl.classList.remove('hidden');
            if (videoFeedEl) videoFeedEl.classList.remove('hidden');
        }
    });
}

function processPrompt() {
    const inputEl = document.getElementById('prompt-input');
    const descEl = document.getElementById('object-description');

    if (!inputEl || !inputEl.value.trim()) return;
    descEl.innerHTML = '<div class="loader"></div>';
    postData("/api/config/prompt", { object_name: inputEl.value }).then(data => console.log("Prompt set response:", data));
}

function stopProcessing() {
    // Clear the prompt in the backend to stop inference
    postData("/api/config/prompt", { object_name: "" }).then(data => {
        console.log("Processing stopped:", data);
        // Reset UI description
        document.getElementById('object-description').textContent = "{Object Description}";
        // Clear input
        document.getElementById('prompt-input').value = "";
        // Disable click select mode if active
        if (document.getElementById('select-object-toggle').checked) {
            document.getElementById('select-object-toggle').checked = false;
            toggleClickSelectMode();
        }
    });
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
    const maskThreshold = parseInt(document.getElementById('mask-slider').value) / 100;
    const displayMode = document.getElementById('display-mode-toggle').checked ? "bounding_box" : "segmentation";
    postData("/api/config/model", { confidence, mask_threshold: maskThreshold, display_mode: displayMode }).then(data => console.log("Model config response:", data));
}

function handleVideoClick(event) {
    if (!frontendState.isClickSelectMode) return;

    const videoFeed = document.getElementById('mock-video-feed');
    const rect = videoFeed.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const normalX = x / rect.width;
    const normalY = y / rect.height;

    const point_prompt = {
        type: "point_prompt",
        points: { x: normalX, y: normalY }
    };

    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(point_prompt));
        document.getElementById('object-description').innerHTML = '<div class="loader"></div>';
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
        postData("/api/config/prompt", { object_name: "" });
    } else {
        videoFeed.classList.remove('cursor-crosshair');
        promptInput.disabled = false;
        promptInput.placeholder = "{Object Name}";
    }
}

// --- 6. DOM Element Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('max-limit').addEventListener('change', updateLimit);
    document.getElementById('sound-toggle').addEventListener('change', updateSoundSetting);
    
    const confidenceSlider = document.getElementById('confidence-slider');
    confidenceSlider.addEventListener('input', () => {
        document.getElementById('confidence-value').textContent = `${confidenceSlider.value}%`;
    });
    confidenceSlider.addEventListener('change', updateModelConfig);

    const maskSlider = document.getElementById('mask-slider');
    maskSlider.addEventListener('input', () => {
        document.getElementById('mask-value').textContent = `${maskSlider.value}%`;
    });
    maskSlider.addEventListener('change', updateModelConfig);
    
    const displayModeToggle = document.getElementById('display-mode-toggle');
    displayModeToggle.addEventListener('change', () => {
        document.getElementById('display-mode-label').textContent = displayModeToggle.checked ? "Bounding Box" : "Segmentation";
        updateModelConfig();
    });

    document.getElementById('select-object-toggle').addEventListener('change', toggleClickSelectMode);
    document.getElementById('mock-video-feed').addEventListener('click', handleVideoClick);

    // Add Stop button listener
    // Check if element exists first to avoid errors if HTML isn't updated yet
    const stopBtn = document.getElementById('stop-processing-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', stopProcessing);
    }

    // Add file input event listeners with null checking
    const videoFileInput = document.getElementById('video-file');
    const imageFileInput = document.getElementById('image-file-input');

    if (videoFileInput) {
        videoFileInput.addEventListener('change', uploadVideo);
        console.log("Video file input listener added successfully");
    } else {
        console.error("Video file input element not found!");
    }

    if (imageFileInput) {
        imageFileInput.addEventListener('change', uploadImage);
        console.log("Image file input listener added successfully");
    } else {
        console.error("Image file input element not found!");
    }

    // Initialize setup functions
    setupSmartPromptDropdown();
    initializeInputModeSwitching();

    // Add video seek listener with null checking
    const videoSeek = document.getElementById('video-seek');
    if (videoSeek) {
        videoSeek.addEventListener('input', (e) => {
            seekVideo(e.target.value);
        });
        console.log("Video seek listener added successfully");
    } else {
        console.error("Video seek element not found!");
    }
});

// --- 7. Dashboard Update Logic ---
function updateDashboard(data) {
    const analytics = data.analytics;
    const countEl = document.getElementById('detected-count');
    const descEl = document.getElementById('object-description');
    const statusBadge = document.getElementById('status-badge');
    const progressBar = document.getElementById('progress-bar');
    const progressLegend = document.getElementById('progress-legend');
    const videoFeed = document.getElementById('mock-video-feed');

    animateValue(countEl, frontendState.currentCount, analytics.count, 500);
    frontendState.currentCount = analytics.count;

    // Handle video metadata updates
    if (analytics.input_mode === 'video') {
        updateVideoProgress(analytics.video_current_frame, analytics.video_total_frames);

        // Update video play/pause button state
        const playPauseBtn = document.getElementById('play-pause-btn');
        if (playPauseBtn) {
            if (analytics.video_playing) {
                playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
            } else {
                playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
            }
        }

        // Update video metadata display
        const currentFrameEl = document.getElementById('current-frame');
        const totalFramesEl = document.getElementById('total-frames');
        const fpsEl = document.getElementById('video-fps');

        if (currentFrameEl) currentFrameEl.textContent = analytics.video_current_frame || 0;
        if (totalFramesEl) totalFramesEl.textContent = analytics.video_total_frames || 0;
        if (fpsEl) fpsEl.textContent = (analytics.video_fps || 0).toFixed(1);
    }

    if(data.video_frame && data.video_frame.startsWith('data:image')) {
       videoFeed.src = data.video_frame;

       // Use class toggling instead of inline styles/setTimeout to prevent layout thrasing
       if (analytics.detected_object && analytics.detected_object !== "N/A") {
           videoFeed.classList.remove('border-transparent');
           videoFeed.classList.add('border-primary');
       } else {
           videoFeed.classList.remove('border-primary');
           videoFeed.classList.add('border-transparent');
       }
    }

    if (analytics.detected_object && analytics.detected_object !== "N/A") {
        const countText = analytics.count > 0 ? `(${analytics.count} found)` : '(searching...)';
        // Only update HTML if it's different to avoid re-rendering loops/flicker
        const newHTML = `Detecting: <span class="font-bold text-primary">${analytics.detected_object}</span> <span class="text-sm text-gray-500">${countText}</span>`;
        if (!descEl.innerHTML.includes(analytics.detected_object)) {
             descEl.innerHTML = newHTML;
             if (analytics.count === 0) {
                descEl.innerHTML += '<div class="loader ml-2" style="display: inline-block; width: 16px; height: 16px; border-width: 2px;"></div>';
            }
        }
    } else if (analytics.detected_object === null || analytics.detected_object === "") {
        descEl.textContent = "{Object Description}";
    }
    
    let percentage = (analytics.count / analytics.max_limit) * 100;
    if (percentage > 100) percentage = 100;
    
    progressBar.style.width = `${percentage}%`;
    if (percentage >= 100) {
        progressBar.className = "bg-red-600 h-3 rounded-full transition-all duration-500";
    } else if (percentage >= 80) {
        progressBar.className = "bg-orange-500 h-3 rounded-full transition-all duration-500";
    } else {
        progressBar.className = "bg-primary h-3 rounded-full transition-all duration-500";
    }

    progressLegend.textContent = `${analytics.count}/${analytics.max_limit}`;
    document.getElementById('max-limit').value = analytics.max_limit;

    statusBadge.textContent = analytics.status;
    const statusColor = analytics.status_color.toLowerCase();
    
    statusBadge.className = "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-all duration-300";
    
    if (statusColor === 'success' || statusColor === 'green' || statusColor === 'approved') {
        statusBadge.classList.add('bg-green-200', 'text-green-800');
    } else if (statusColor === 'orange' || statusColor === 'waiting') {
        statusBadge.classList.add('bg-orange-200', 'text-orange-800');
    } else {
        statusBadge.classList.add('bg-red-200', 'text-red-800', 'animate-pulse');
    }

    if (analytics.trigger_sound) {
        triggerNotification();
        document.body.style.backgroundColor = '#fef2f2';
        setTimeout(() => { document.body.style.backgroundColor = '#F8F9FA'; }, 200);
    }
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


// --- 9. Smart Prompt Dropdown Functions ---
function setupSmartPromptDropdown() {
    const promptInput = document.getElementById('prompt-input');
    const dropdownToggle = document.getElementById('dropdown-toggle');
    const suggestionsPanel = document.getElementById('object-suggestions');

    dropdownToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHidden = suggestionsPanel.classList.contains('hidden');
        if (isHidden) {
            suggestionsPanel.classList.remove('hidden');
            const inputRect = promptInput.getBoundingClientRect();
            suggestionsPanel.style.width = '100%';
            suggestionsPanel.style.left = '0';
            suggestionsPanel.style.top = `${inputRect.height + 4}px`;
            suggestionsPanel.style.maxWidth = 'none';
        } else {
            suggestionsPanel.classList.add('hidden');
        }
    });

    document.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', (e) => {
            promptInput.value = e.target.textContent.trim();
            suggestionsPanel.classList.add('hidden');
            promptInput.focus();
        });
    });

    document.addEventListener('click', () => suggestionsPanel.classList.add('hidden'));
    suggestionsPanel.addEventListener('click', (e) => e.stopPropagation());
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') suggestionsPanel.classList.add('hidden');
    });
}

// --- 10. Input Mode Management Functions ---
function initializeInputModeSwitching() {
    console.log("Initializing Input Mode Switching...");
    
    // Initialize with RTSP mode (default)
    switchInputMode('rtsp');

    // Set up event listeners for radio buttons
    document.querySelectorAll('input[name="input-mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.checked) switchInputMode(e.target.value);
        });
    });
    
    // Setup Shared Drop Zone Logic
    const dropZone = document.getElementById('display-zone');
    const overlay = document.getElementById('upload-overlay');
    
    if (!overlay) console.error("Upload overlay element NOT found!");
    if (!dropZone) console.error("Display zone element NOT found!");

    // Forward clicks on overlay to the correct file input
    if (overlay) {
        // Make sure overlay is clickable and visible
        overlay.style.cursor = 'pointer';
        overlay.style.zIndex = '50';

        overlay.addEventListener('click', (e) => {
            console.log("Overlay clicked!");
            e.preventDefault();
            e.stopPropagation();

            const modeInput = document.querySelector('input[name="input-mode"]:checked');
            if (!modeInput) {
                console.error("No input mode selected!");
                alert("Please select an input mode first (Video or Image)");
                return;
            }

            const mode = modeInput.value;
            console.log(`Current mode: ${mode}`);

            if (mode === 'video') {
                const fileInput = document.getElementById('video-file');
                console.log("Triggering Video Input click", fileInput);
                if (fileInput) {
                    fileInput.accept = 'video/*';
                    fileInput.click();
                } else {
                    console.error("Video file input not found!");
                    alert("Video file input not found. Please refresh the page.");
                }
            } else if (mode === 'image') {
                const fileInput = document.getElementById('image-file-input');
                console.log("Triggering Image Input click", fileInput);
                if (fileInput) {
                    fileInput.accept = 'image/*';
                    fileInput.click();
                } else {
                    console.error("Image file input not found!");
                    alert("Image file input not found. Please refresh the page.");
                }
            } else if (mode === 'rtsp') {
                console.log("RTSP mode - no file upload needed");
                alert("RTSP mode uses stream input, not file upload. Please enter an RTSP URL.");
            }
        });
    }

    // Handle file selection
    const videoFileInput = document.getElementById('video-file');
    if (videoFileInput) {
        videoFileInput.addEventListener('change', (e) => {
            console.log("Video file selected:", e.target.files[0]);
            uploadVideo();
        });
    }
    
    const imageFileInput = document.getElementById('image-file-input');
    if (imageFileInput) {
        imageFileInput.addEventListener('change', (e) => {
            console.log("Image file selected:", e.target.files[0]);
            uploadImage();
        });
    }

    // Drag & Drop Support
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-primary', 'bg-primary/5');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-primary', 'bg-primary/5');
        });
        
        dropZone.addEventListener('drop', (e) => {
            try {
                e.preventDefault();
                dropZone.classList.remove('border-primary', 'bg-primary/5');
                console.log("File dropped! Files count:", e.dataTransfer.files.length);

                // Get current mode with null checking
                const modeInput = document.querySelector('input[name="input-mode"]:checked');
                if (!modeInput) {
                    console.error("No input mode selected when file dropped");
                    alert("Please select an input mode first (RTSP, Video, or Image)");
                    return;
                }

                const mode = modeInput.value;
                console.log("Current mode:", mode);

                if (e.dataTransfer.files.length === 0) {
                    console.log("No files in drop event");
                    return;
                }

                const file = e.dataTransfer.files[0];
                console.log("Dropped file:", file.name, file.type, file.size);

                if (mode === 'video') {
                    if (file.type.startsWith('video/')) {
                        console.log("Processing video drop");
                        if (videoFileInput) {
                            // Directly call uploadVideo with the file
                            const tempInput = document.createElement('input');
                            tempInput.type = 'file';
                            tempInput.files = e.dataTransfer.files;

                            // Simulate the file selection
                            videoFileInput.files = e.dataTransfer.files;

                            console.log("Video file assigned to input, calling uploadVideo()");
                            uploadVideo();
                        } else {
                            console.error("Video file input element not found!");
                            alert("Video file input not found. Please refresh the page.");
                        }
                    } else {
                        console.warn("Invalid file type for video mode:", file.type);
                        alert(`Please drop a valid video file (MP4, AVI, MOV, etc.). Current file: ${file.type}`);
                    }
                } else if (mode === 'image') {
                    if (file.type.startsWith('image/')) {
                        console.log("Processing image drop");
                        if (imageFileInput) {
                            // Directly call uploadImage with the file
                            const tempInput = document.createElement('input');
                            tempInput.type = 'file';
                            tempInput.files = e.dataTransfer.files;

                            // Simulate the file selection
                            imageFileInput.files = e.dataTransfer.files;

                            console.log("Image file assigned to input, calling uploadImage()");
                            uploadImage();
                        } else {
                            console.error("Image file input element not found!");
                            alert("Image file input not found. Please refresh the page.");
                        }
                    } else {
                        console.warn("Invalid file type for image mode:", file.type);
                        alert(`Please drop a valid image file (JPEG, PNG, etc.). Current file: ${file.type}`);
                    }
                } else {
                    console.warn("Drop not supported in mode:", mode);
                    alert(`File upload is not supported in ${mode} mode. Please select Video or Image mode.`);
                }
            } catch (error) {
                console.error("Error handling file drop:", error);
                alert("Error processing dropped file: " + error.message);
            }
        });
    }
}

async function switchInputMode(mode) {
    console.log(`Switching to mode: ${mode}`);

    // 1. Hide all Control Panels
    document.getElementById('rtsp-panel').classList.add('hidden');
    document.getElementById('video-panel').classList.add('hidden');
    document.getElementById('image-panel').classList.add('hidden');

    // 2. Show Selected Control Panel
    const targetPanel = document.getElementById(`${mode}-panel`);
    if (targetPanel) targetPanel.classList.remove('hidden');
    
    // 3. Update Display Zone Placeholder & Overlay
    const placeholderHtml = document.getElementById('stream-placeholder');
    const overlay = document.getElementById('upload-overlay');
    const streamFeed = document.getElementById('mock-video-feed');
    
    if (!placeholderHtml) return;

    if (mode === 'rtsp') {
        placeholderHtml.innerHTML = `
            <i class="fa-regular fa-circle-play text-gray-400 text-6xl mb-4"></i>
            <p class="text-gray-400 font-medium">&lt;Camera RTSP Live Stream&gt;</p>
        `;
        if (overlay) overlay.classList.add('hidden'); // Disable click-to-upload
        placeholderHtml.classList.remove('hidden');
        if (streamFeed) streamFeed.classList.add('hidden');
    } 
    else if (mode === 'video') {
        placeholderHtml.innerHTML = `
            <p class="text-gray-400 text-lg mb-2">Drop Video File Here</p>
            <p class="text-gray-400 text-sm mb-4">or</p>
            <button onclick="document.getElementById('video-file').click()" class="bg-primary hover:bg-blue-800 text-white font-bold px-6 py-2 rounded-lg text-sm transition-colors shadow-md pointer-events-auto relative z-50">
                <i class="fa-solid fa-folder-open mr-2"></i>
                Browse Files
            </button>
        `;
        // Show overlay only if no file is loaded yet
        const infoPanel = document.getElementById('video-file-info');
        const isInfoHidden = infoPanel && infoPanel.classList.contains('hidden');
        console.log(`Video mode: Info hidden? ${isInfoHidden}`);
        
        if (isInfoHidden) {
             if (overlay) overlay.classList.add('hidden'); // Hide overlay so button is clickable
             placeholderHtml.classList.remove('hidden');
             if (streamFeed) streamFeed.classList.add('hidden');
             // Re-enable overlay for drag/drop on the container background if needed, 
             // but for now, let's rely on the button and the drop zone event listeners.
             // To make the whole area clickable again without blocking the button, we'd need complex layering.
             // The user specifically asked for the button.
        } else {
             // File already loaded, show feed
             if (overlay) overlay.classList.add('hidden');
        }
    } 
    else if (mode === 'image') {
        placeholderHtml.innerHTML = `
            <p class="text-gray-400 text-lg mb-2">Drop Image File Here</p>
            <p class="text-gray-400 text-sm mb-4">or</p>
            <button onclick="document.getElementById('image-file-input').click()" class="bg-primary hover:bg-blue-800 text-white font-bold px-6 py-2 rounded-lg text-sm transition-colors shadow-md pointer-events-auto relative z-50">
                <i class="fa-solid fa-folder-open mr-2"></i>
                Browse Files
            </button>
        `;
        // Show overlay only if no file is loaded yet
        const infoPanel = document.getElementById('image-file-info');
        const isInfoHidden = infoPanel && infoPanel.classList.contains('hidden');
        console.log(`Image mode: Info hidden? ${isInfoHidden}`);

        if (isInfoHidden) {
             if (overlay) overlay.classList.add('hidden'); // Hide overlay so button is clickable
             placeholderHtml.classList.remove('hidden');
             if (streamFeed) streamFeed.classList.add('hidden');
        } else {
             if (overlay) overlay.classList.add('hidden');
        }
    }

    // 4. Notify Backend
    try {
        await postData('/api/config/input-mode', { mode: mode });
    } catch (error) {
        console.error('Error switching input mode:', error);
    }
}

async function setRtspUrl() {
    const url = document.getElementById('rtsp-url-input').value;
    console.log("RTSP URL input:", url);

    // Support device '0' for local webcam
    if (!url.trim()) {
        // Auto-fill with device 0 if empty
        document.getElementById('rtsp-url-input').value = '0';
        const deviceUrl = '0';
        console.log("Auto-setting device to:", deviceUrl);

        document.getElementById('stream-placeholder').innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to local device...</p>';

        try {
            const response = await postData("/api/config/stream", { url: deviceUrl });
            if (response.status === 'success') {
                document.getElementById('stream-placeholder').classList.add('hidden');
                document.getElementById('live-indicator').classList.remove('hidden');
                document.getElementById('upload-overlay').classList.add('hidden'); // Disable drop zone for RTSP
                console.log("Device connection successful");
            }
        } catch (error) {
            console.error("Error connecting to device:", error);
            alert("Failed to connect to device. Please check device permissions.");
            switchInputMode('rtsp');
        }
        return;
    }

    // Handle URL format with device parameter
    let processedUrl = url.trim();

    // Convert device '0' to proper format
    if (processedUrl === '0') {
        processedUrl = '0'; // Keep as device index
        document.getElementById('stream-placeholder').innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to local device (webcam)...</p>';
    } else if (!processedUrl.startsWith('rtsp://') && processedUrl !== '0') {
        // Auto-add rtsp:// prefix if missing
        if (!processedUrl.includes('://')) {
            processedUrl = 'rtsp://' + processedUrl;
        }
        document.getElementById('stream-placeholder').innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to RTSP stream...</p>';
    }

    console.log("Processed URL:", processedUrl);

    try {
        const response = await postData("/api/config/stream", { url: processedUrl });
        if (response.status === 'success') {
            document.getElementById('stream-placeholder').classList.add('hidden');
            document.getElementById('live-indicator').classList.remove('hidden');
            document.getElementById('upload-overlay').classList.add('hidden'); // Disable drop zone for RTSP
            document.getElementById('mock-video-feed').classList.remove('hidden');
            console.log("RTSP connection successful");
        } else {
            alert('Failed to connect to RTSP: ' + response.message);
            switchInputMode('rtsp'); // Reset UI
        }
    } catch (error) {
        console.error('RTSP connection error:', error);
        alert('Error connecting to RTSP. Please check if the backend is running.');
        switchInputMode('rtsp');
    }
}

async function uploadVideo() {
    try {
        console.log("uploadVideo() function called");

        const fileInput = document.getElementById('video-file');
        if (!fileInput) {
            console.error("Video file input element not found!");
            alert("Video file input not found. Please refresh the page.");
            return;
        }

        const file = fileInput.files[0];
        if (!file) {
            console.log("No video file selected");
            return;
        }

        // Validate file type
        if (!file.type.startsWith('video/')) {
            alert("Please select a valid video file");
            return;
        }

        console.log("Uploading video file:", file.name, file.type, file.size);

        // Show loading in placeholder
        const streamPlaceholder = document.getElementById('stream-placeholder');
        if (streamPlaceholder) {
            streamPlaceholder.innerHTML = `
                <div class="loader"></div><p class="mt-2 text-sm text-gray-500">Uploading ${file.name}...</p>
            `;
        }

        const formData = new FormData();
        formData.append('file', file);

        console.log("Sending video upload request to:", `${API_BASE_URL}/api/upload/video`);

        const response = await fetch(`${API_BASE_URL}/api/upload/video`, {
            method: 'POST',
            body: formData
        });

        console.log("Video upload response status:", response.status);

        const result = await response.json();
        console.log("Video upload response data:", result);

        if (response.ok && result.status === 'success') {
            // 1. Update Control Panel (Show File Info)
            const videoFilename = document.getElementById('video-filename');
            const videoFileInfo = document.getElementById('video-file-info');
            const videoPlaybackControls = document.getElementById('video-playback-controls');

            if (videoFilename) videoFilename.textContent = file.name;
            if (videoFileInfo) videoFileInfo.classList.remove('hidden');
            if (videoPlaybackControls) videoPlaybackControls.classList.remove('hidden');

            // 2. Update Display Zone
            const uploadOverlay = document.getElementById('upload-overlay');
            if (uploadOverlay) uploadOverlay.classList.add('hidden'); // Disable drop zone

            if (streamPlaceholder) streamPlaceholder.classList.add('hidden');

            const mockVideoFeed = document.getElementById('mock-video-feed');
            if (mockVideoFeed) mockVideoFeed.classList.remove('hidden');

            updateVideoMetadata(result);
            await switchInputMode('video'); // Refresh backend state

            console.log("Video upload successful");
        } else {
            console.error("Video upload failed:", result);
            alert('Failed to upload video: ' + (result.message || 'Unknown error'));
            switchInputMode('video'); // Reset
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        alert('Error uploading video: ' + error.message);

        // Reset UI on error
        try {
            await switchInputMode('video');
        } catch (resetError) {
            console.error("Error resetting video mode:", resetError);
        }
    }
}

async function clearVideo() {
    // Reset backend
    await switchInputMode('video'); 
    
    // Reset UI
    document.getElementById('video-file-info').classList.add('hidden');
    document.getElementById('video-playback-controls').classList.add('hidden');
    document.getElementById('video-file').value = '';
    
    // Reset Display Zone by calling switchInputMode again which checks the 'hidden' class
    switchInputMode('video');
}

async function uploadImage() {
    try {
        console.log("uploadImage() function called");

        const fileInput = document.getElementById('image-file-input');
        if (!fileInput) {
            console.error("Image file input element not found!");
            alert("Image file input not found. Please refresh the page.");
            return;
        }

        const file = fileInput.files[0];
        if (!file) {
            console.log("No image file selected");
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert("Please select a valid image file");
            return;
        }

        console.log("Uploading image file:", file.name, file.type, file.size);

        // Show loading in placeholder
        const streamPlaceholder = document.getElementById('stream-placeholder');
        if (streamPlaceholder) {
            streamPlaceholder.innerHTML = `
                <div class="loader"></div><p class="mt-2 text-sm text-gray-500">Uploading ${file.name}...</p>
            `;
        }

        const formData = new FormData();
        formData.append('file', file);

        console.log("Sending image upload request to:", `${API_BASE_URL}/api/upload/image`);

        const response = await fetch(`${API_BASE_URL}/api/upload/image`, {
            method: 'POST',
            body: formData
        });

        console.log("Image upload response status:", response.status);

        const result = await response.json();
        console.log("Image upload response data:", result);

        if (response.ok && result.status === 'success') {
            // 1. Update Control Panel
            const imageFilename = document.getElementById('image-filename');
            const imageFileInfo = document.getElementById('image-file-info');

            if (imageFilename) imageFilename.textContent = file.name;
            if (imageFileInfo) imageFileInfo.classList.remove('hidden');

            // 2. Update Display Zone (Preview local file immediately or wait for websocket)
            try {
                displayUploadedImage(file);
            } catch (displayError) {
                console.error("Error displaying uploaded image:", displayError);
            }

            const uploadOverlay = document.getElementById('upload-overlay');
            if (uploadOverlay) uploadOverlay.classList.add('hidden');

            if (streamPlaceholder) streamPlaceholder.classList.add('hidden');

            console.log("Image upload successful");
        } else {
            console.error("Image upload failed:", result);
            alert('Failed to upload image: ' + (result.message || 'Unknown error'));
            switchInputMode('image');
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        alert('Error uploading image: ' + error.message);

        // Reset UI on error
        try {
            await switchInputMode('image');
        } catch (resetError) {
            console.error("Error resetting image mode:", resetError);
        }
    }
}

function displayUploadedImage(file) {
    try {
        console.log("Displaying uploaded image:", file.name);

        const reader = new FileReader();
        reader.onload = (e) => {
            const videoFeed = document.getElementById('mock-video-feed');
            if (!videoFeed) {
                console.error("Mock video feed element not found for image display!");
                return;
            }

            console.log("Setting image source to video feed element");
            videoFeed.src = e.target.result;
            videoFeed.classList.remove('hidden');

            // For images, we should use img element properties
            videoFeed.alt = `Uploaded image: ${file.name}`;
        };
        reader.readAsDataURL(file);
    } catch (error) {
        console.error("Error displaying uploaded image:", error);
        alert("Error displaying uploaded image: " + error.message);
    }
}

async function clearUploadedImage() {
    try {
        console.log("Clearing uploaded image...");

        const response = await postData("/api/config/clear-image", {});
        if (response.status === 'success') {
            console.log("Image clear response successful");

            // Reset UI with null checking
            const imageFileInfo = document.getElementById('image-file-info');
            const imageFileInput = document.getElementById('image-file-input');
            const mockVideoFeed = document.getElementById('mock-video-feed');
            const streamPlaceholder = document.getElementById('stream-placeholder');

            if (imageFileInfo) {
                imageFileInfo.classList.add('hidden');
                console.log("Hidden image file info panel");
            }

            if (imageFileInput) {
                imageFileInput.value = '';
                console.log("Cleared image file input");
            }

            // Reset display
            if (mockVideoFeed) {
                mockVideoFeed.classList.add('hidden');
                mockVideoFeed.src = '';
                mockVideoFeed.alt = '';
                console.log("Hidden mock video feed");
            }

            if (streamPlaceholder) {
                streamPlaceholder.classList.remove('hidden');
                console.log("Shown stream placeholder");
            }

            console.log('Image cleared successfully');
            switchInputMode('image');
        }
    } catch (error) {
        console.error('Error clearing uploaded image:', error);
    }
}

function updateVideoProgress(currentFrame, totalFrames) {
    const videoSeek = document.getElementById('video-seek');
    const frameCounter = document.getElementById('frame-counter');

    if (videoSeek && frameCounter) {
        videoSeek.value = currentFrame || 0;
        frameCounter.textContent = `${currentFrame || 0}/${totalFrames || 0}`;
    }
}

// --- Video Control Functions ---
async function seekVideo(frameIndex) {
    try {
        const response = await postData("/api/config/video/seek", { frame: parseInt(frameIndex) });
        console.log("Video seek response:", response);
    } catch (error) {
        console.error("Error seeking video:", error);
    }
}

async function toggleVideoPlayback() {
    try {
        console.log("toggleVideoPlayback() function called");

        const playPauseBtn = document.getElementById('play-pause-btn');
        if (!playPauseBtn) {
            console.error("Play/pause button not found");
            return;
        }

        const currentState = !playPauseBtn.innerHTML.includes('fa-play');
        console.log("Current video state:", currentState ? "playing" : "paused");

        const response = await postData("/api/config/video/play-pause", { playing: !currentState });
        console.log("Video playback toggle response:", response);

        // Update button icon immediately for better UX
        if (response && response.status === 'success') {
            const isPlaying = response.data && response.data.playing;
            console.log("New playing state:", isPlaying);

            playPauseBtn.innerHTML = isPlaying ?
                '<i class="fa-solid fa-pause"></i>' :
                '<i class="fa-solid fa-play"></i>';
        }
    } catch (error) {
        console.error("Error toggling video playback:", error);
    }
}

function updateVideoMetadata(metadata) {
    if (!metadata) {
        console.error("No metadata provided to updateVideoMetadata");
        return;
    }

    // Update video seek slider
    if (metadata.total_frames) {
        const videoSeek = document.getElementById('video-seek');
        if (videoSeek) {
            videoSeek.max = metadata.total_frames;
            videoSeek.value = 0;
            console.log("Updated video seek slider max frames:", metadata.total_frames);
        }
    }

    // Update FPS display
    if (metadata.fps) {
        const fpsEl = document.getElementById('video-fps');
        if (fpsEl) {
            fpsEl.textContent = metadata.fps.toFixed(1);
        }
    }

    // Update frame counter
    updateVideoProgress(0, metadata.total_frames || 0);

    console.log("Video metadata updated:", metadata);
}

// --- RTSP Stop Streaming Function ---
async function stopRtspStreaming() {
    try {
        console.log("stopRtspStreaming() function called");

        const rtspInput = document.getElementById('rtsp-url-input');
        if (!rtspInput) {
            console.error("RTSP URL input element not found!");
            return;
        }

        const currentUrl = rtspInput.value;
        console.log("Stopping RTSP streaming for URL:", currentUrl);

        // Show loading indicator
        const streamPlaceholder = document.getElementById('stream-placeholder');
        if (streamPlaceholder) {
            streamPlaceholder.innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Stopping stream...</p>';
        }

        // Call backend API to stop streaming
        const response = await postData("/api/config/stream", { url: "" });
        if (response.status === 'success') {
            console.log("RTSP stream stopped successfully");

            // Reset UI
            if (streamPlaceholder) {
                streamPlaceholder.innerHTML = `
                    <div class="text-center text-gray-500">
                        <i class="fa-solid fa-video text-4xl mb-3"></i>
                        <p class="font-medium">Waiting for RTSP input...</p>
                        <p class="text-sm text-gray-400">Enter RTSP URL or use '0' for webcam</p>
                    </div>
                `;
            }

            const uploadOverlay = document.getElementById('upload-overlay');
            if (uploadOverlay) {
                uploadOverlay.classList.remove('hidden');
            }

            const liveIndicator = document.getElementById('live-indicator');
            if (liveIndicator) {
                liveIndicator.classList.add('hidden');
            }

            const mockVideoFeed = document.getElementById('mock-video-feed');
            if (mockVideoFeed) {
                mockVideoFeed.classList.add('hidden');
            }

            // Clear RTSP URL input
            rtspInput.value = '';
        } else {
            console.error("Failed to stop RTSP stream:", response);
            alert('Failed to stop RTSP stream: ' + (response.message || 'Unknown error'));
        }
    } catch (error) {
        console.error("Error stopping RTSP stream:", error);
        alert('Error stopping RTSP stream: ' + error.message);
    }
}

