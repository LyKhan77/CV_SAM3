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

// --- 5. UI Event Handlers ---
function activateStream() {
    // ... (existing logic)
}

function processPrompt() {
    const inputEl = document.getElementById('prompt-input');
    const descPromptEl = document.getElementById('desc-prompt');

    if (!inputEl || !inputEl.value.trim()) return;
    
    // Just update the UI and send to backend storage, don't trigger run
    descPromptEl.textContent = inputEl.value;
    postData("/api/config/prompt", { object_name: inputEl.value }).then(data => {
        console.log("Prompt stored:", data);
    });
}

async function runSegmentation() {
    const btn = document.getElementById('run-segmentation-btn');
    const statusEl = document.getElementById('desc-status');
    const statusDot = document.getElementById('status-dot');
    
    // UI Loading State
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="loader w-4 h-4 border-white border-t-transparent"></div> Processing...';
    }
    
    // Lock sliders
    document.getElementById('confidence-slider').disabled = true;
    document.getElementById('mask-slider').disabled = true;
    
    if (statusEl) statusEl.textContent = "Processing...";
    if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-yellow-400 animate-pulse";

    try {
        const response = await postData("/api/config/run", {});
        console.log("Run response:", response);
        // Button re-enabled by WebSocket status update eventually
    } catch (e) {
        console.error("Run failed:", e);
        // Reset on error
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Segmentation';
        }
    }
}

async function clearMask() {
    try {
        await postData("/api/config/clear-mask", {});
        
        // Clear Visuals
        const descStatusEl = document.getElementById('desc-status');
        const statusDot = document.getElementById('status-dot');
        
        if (descStatusEl) descStatusEl.textContent = "Ready";
        if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-gray-400";
        
        // Unlock sliders
        document.getElementById('confidence-slider').disabled = false;
        document.getElementById('mask-slider').disabled = false;

    } catch (e) {
        console.error("Clear mask failed:", e);
    }
}

// ... (stopProcessing removed as requested)

function updateLimit() {
    const limit = parseInt(document.getElementById('max-limit').value) || 100;
    postData("/api/config/limit", { value: limit }).then(data => console.log("Limit set response:", data));
}

// ... (rest of functions)

// --- 6. Model Configuration Logic ---
function updateModelConfig() {
    const confidence = parseInt(document.getElementById('confidence-slider').value) / 100;
    const maskThreshold = parseInt(document.getElementById('mask-slider').value) / 100;

    postData("/api/config/model", { 
        confidence: confidence, 
        mask_threshold: maskThreshold 
    }).then(data => {
        console.log("Model config updated:", data);
    });
}

function setupModelConfigListeners() {
    const confSlider = document.getElementById('confidence-slider');
    const maskSlider = document.getElementById('mask-slider');
    const confValue = document.getElementById('confidence-value');
    const maskValue = document.getElementById('mask-value');

    if (confSlider && confValue) {
        confSlider.addEventListener('input', (e) => {
            confValue.textContent = `${e.target.value}%`;
        });
        confSlider.addEventListener('change', updateModelConfig);
    }

    if (maskSlider && maskValue) {
        maskSlider.addEventListener('input', (e) => {
            maskValue.textContent = `${e.target.value}%`;
        });
        maskSlider.addEventListener('change', updateModelConfig);
    }
}

// --- 6.1 Summary Panel Listeners ---
function updateSound() {
    const enabled = document.getElementById('sound-toggle').checked;
    postData("/api/config/sound", { enabled: enabled }).then(data => {
        console.log("Sound config updated:", data);
    });
}

function updateLimit() {
    const limitInput = document.getElementById('max-limit');
    const limit = parseInt(limitInput.value) || 100;
    console.log("Updating limit to:", limit);
    
    // Optimistic UI update for progress legend (optional, but good for responsiveness)
    const progressLegend = document.getElementById('progress-legend');
    if (progressLegend) {
        const currentText = progressLegend.textContent.split('/')[0]; // Get current count
        progressLegend.textContent = `${currentText}/${limit}`;
    }

    postData("/api/config/limit", { value: limit }).then(data => console.log("Limit set response:", data));
}

function setupSummaryListeners() {
    const limitInput = document.getElementById('max-limit');
    const soundToggle = document.getElementById('sound-toggle');

    if (limitInput) {
        // Use 'input' for real-time updates, or 'change' for commit-on-blur. 
        // 'change' is safer for API calls, 'input' might flood. 
        // Let's stick to 'change' but ensure it works. 
        // Actually, user said "set the Max Limit", implying they finished setting it.
        limitInput.addEventListener('change', updateLimit);
        limitInput.addEventListener('blur', updateLimit); // Ensure blur also triggers
    }
    
    if (soundToggle) {
        soundToggle.addEventListener('change', updateSound);
    }
    console.log("Summary listeners setup complete.");
}

// --- 7. Dashboard Update Logic ---
function updateDashboard(data) {
    const analytics = data.analytics;
    const countEl = document.getElementById('detected-count');
    
    // Update new status fields
    const descPromptEl = document.getElementById('desc-prompt');
    const descStatusEl = document.getElementById('desc-status');
    const statusDot = document.getElementById('status-dot');
    const runBtn = document.getElementById('run-segmentation-btn');

    animateValue(countEl, frontendState.currentCount, analytics.count, 500);
    frontendState.currentCount = analytics.count;
    
    // Update Status UI from Backend State
    if (descStatusEl && analytics.process_status) {
        descStatusEl.textContent = analytics.process_status;
        
        if (analytics.process_status === "Processing...") {
            if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-yellow-400 animate-pulse";
        } else if (analytics.process_status === "Done") {
            if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-green-500";
            // Re-enable Run button and sliders
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Segmentation';
            }
            document.getElementById('confidence-slider').disabled = false;
            document.getElementById('mask-slider').disabled = false;
        } else {
             if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-gray-400";
             // Ready state
             if (runBtn) {
                runBtn.disabled = false;
                runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Segmentation';
             }
             document.getElementById('confidence-slider').disabled = false;
             document.getElementById('mask-slider').disabled = false;
        }
    }
    
    if (descPromptEl && analytics.detected_object !== "N/A") {
        descPromptEl.textContent = analytics.detected_object;
    }

    // --- Updated Video Feed (Overlay) ---
    const videoFeed = document.getElementById('mock-video-feed');
    const placeholder = document.getElementById('stream-placeholder');
    if (videoFeed && data.video_frame) {
        videoFeed.src = data.video_frame;
        videoFeed.classList.remove('hidden');
        if (placeholder) placeholder.classList.add('hidden');
    }

    // --- Update Status Badge ---
    const statusBadge = document.getElementById('status-badge');
    if (statusBadge && analytics.status) {
        statusBadge.textContent = analytics.status;
        // Reset classes
        statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-colors";
        
        if (analytics.status === "Approved") {
            // "Approved" in this context implies condition met (Count >= Limit)
            // User requested Sound Alert on this condition, so it might be an "Alert" state.
            // However, naming "Approved" usually implies Success. 
            // Let's use Red for Alert/Limit Reached to match the sound urgency.
            statusBadge.classList.add("bg-red-200", "text-red-800");
        } else {
            statusBadge.classList.add("bg-yellow-100", "text-yellow-800");
        }
    }

    // --- Update Progress Bar ---
    const progressBar = document.getElementById('progress-bar');
    const progressLegend = document.getElementById('progress-legend');
    
    if (progressBar && analytics.max_limit > 0) {
        const percentage = Math.min((analytics.count / analytics.max_limit) * 100, 100);
        progressBar.style.width = `${percentage}%`;
        
        // Color logic: Red if over limit, Primary Blue otherwise
        if (analytics.count >= analytics.max_limit) {
            progressBar.className = "h-2 rounded-full bg-red-600 transition-all duration-500";
        } else {
            progressBar.className = "h-2 rounded-full bg-primary transition-all duration-500";
        }
    }
    
    if (progressLegend) {
        progressLegend.textContent = `${analytics.count}/${analytics.max_limit}`;
    }

    // --- Sound Alert ---
    if (analytics.trigger_sound) {
        triggerNotification();
    }

    // Sync Object List Clearing
    // If detected object count goes to 0 and status is Ready (cleared), clear the list visual if needed?
    // Actually, user requested explicit clearing on Image Remove.
}

// ... (rest of file)
function openInputHelpModal() {
    const modal = document.getElementById('input-help-modal');
    const content = document.getElementById('input-help-modal-content');
    if (!modal || !content) return;
    
    modal.classList.remove('hidden');
    setTimeout(() => {
        modal.classList.remove('opacity-0');
        content.classList.remove('scale-95');
        content.classList.add('scale-100');
    }, 10);
}

function closeInputHelpModal() {
    const modal = document.getElementById('input-help-modal');
    const content = document.getElementById('input-help-modal-content');
    if (!modal || !content) return;

    modal.classList.add('opacity-0');
    content.classList.remove('scale-100');
    content.classList.add('scale-95');
    
    setTimeout(() => {
        modal.classList.add('hidden');
    }, 300);
}

// Add global click listener for modal closing
document.addEventListener('DOMContentLoaded', () => {
    const helpModal = document.getElementById('help-modal');
    if (helpModal) {
        helpModal.addEventListener('click', (e) => {
            if (e.target.id === 'help-modal') closeHelpModal();
        });
    }
    
    const inputHelpModal = document.getElementById('input-help-modal');
    if (inputHelpModal) {
        inputHelpModal.addEventListener('click', (e) => {
            if (e.target.id === 'input-help-modal') closeInputHelpModal();
        });
    }
});

// Update switchInputMode to clear object list
async function switchInputMode(mode) {
    // ... (existing code)
    
    // Clear object list on mode switch
    const listContainer = document.getElementById('object-list-container');
    if (listContainer) {
        listContainer.innerHTML = '<p class="col-span-full text-center text-sm">No objects saved yet.</p>';
    }

    // ...
}

// Update clearUploadedImage to clear object list
async function clearUploadedImage() {
    // ... (existing code)
    const listContainer = document.getElementById('object-list-container');
    if (listContainer) {
        listContainer.innerHTML = '<p class="col-span-full text-center text-sm">No objects saved yet.</p>';
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
    
    if (overlay) {
        // Make sure overlay is clickable and visible
        overlay.style.cursor = 'pointer';
        overlay.style.zIndex = '50';

        // Remove old listeners if any (by cloning) - optional but good practice if re-init
        // For now, just add listener
        overlay.onclick = (e) => {
            console.log("Overlay clicked!");
            e.preventDefault();
            e.stopPropagation();

            const modeInput = document.querySelector('input[name="input-mode"]:checked');
            if (!modeInput) return;

            const mode = modeInput.value;
            console.log(`Overlay Click - Current mode: ${mode}`);

            if (mode === 'video') {
                const fileInput = document.getElementById('video-file');
                if (fileInput) fileInput.click();
            } else if (mode === 'image') {
                const fileInput = document.getElementById('image-file-input');
                if (fileInput) fileInput.click();
            }
        };
    }

    // Re-attach change listeners to inputs directly to be safe
    const videoFileInput = document.getElementById('video-file');
    if (videoFileInput) {
        videoFileInput.onchange = (e) => {
            console.log("Video Input Change");
            uploadVideo();
        };
    }
    
    const imageFileInput = document.getElementById('image-file-input');
    if (imageFileInput) {
        imageFileInput.onchange = (e) => {
            console.log("Image Input Change");
            uploadImage();
        };
    }

    // Drag & Drop Support (Keep existing logic but ensure it works)
    if (dropZone) {
        dropZone.ondragover = (e) => {
            e.preventDefault();
            dropZone.classList.add('border-primary', 'bg-primary/5');
        };
        
        dropZone.ondragleave = () => {
            dropZone.classList.remove('border-primary', 'bg-primary/5');
        };
        
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-primary', 'bg-primary/5');
            
            const modeInput = document.querySelector('input[name="input-mode"]:checked');
            if (!modeInput) return;
            const mode = modeInput.value;

            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (mode === 'video' && videoFileInput) {
                    videoFileInput.files = e.dataTransfer.files;
                    uploadVideo();
                } else if (mode === 'image' && imageFileInput) {
                    imageFileInput.files = e.dataTransfer.files;
                    uploadImage();
                }
            }
        };
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

    // Clear Object List
    const listContainer = document.getElementById('object-list-container');
    if (listContainer) {
        listContainer.innerHTML = '<p class="col-span-full text-center text-sm">No objects saved yet.</p>';
    }
    
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

            // streamPlaceholder hiding is handled by displayUploadedImage, but double check here
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
            const placeholder = document.getElementById('stream-placeholder');
            
            if (!videoFeed) {
                console.error("Mock video feed element not found for image display!");
                return;
            }

            console.log("Setting image source to video feed element");
            videoFeed.src = e.target.result;
            videoFeed.classList.remove('hidden');
            videoFeed.classList.remove('absolute'); // Remove absolute if it conflicts
            videoFeed.classList.add('relative'); // Make it flow
            
            if (placeholder) {
                placeholder.classList.add('hidden');
            }

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

            console.log('Image cleared successfully. Clearing object list...');
            
            // Clear Object List
            const listContainer = document.getElementById('object-list-container');
            if (listContainer) {
                listContainer.innerHTML = '<p class="col-span-full text-center text-sm">No objects saved yet.</p>';
                console.log("Object list cleared.");
            } else {
                console.error("Object list container not found!");
            }
            
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

// --- 11. Help Modal Functions ---
function openHelpModal() {
    const modal = document.getElementById('help-modal');
    const content = document.getElementById('help-modal-content');
    if (!modal || !content) return;
    
    modal.classList.remove('hidden');
    // Small delay to allow display:block to apply before opacity transition
    setTimeout(() => {
        modal.classList.remove('opacity-0');
        content.classList.remove('scale-95');
        content.classList.add('scale-100');
    }, 10);
}

function closeHelpModal() {
    const modal = document.getElementById('help-modal');
    const content = document.getElementById('help-modal-content');
    if (!modal || !content) return;

    modal.classList.add('opacity-0');
    content.classList.remove('scale-100');
    content.classList.add('scale-95');
    
    setTimeout(() => {
        modal.classList.add('hidden');
    }, 300);
}

// --- 12. Export & Snapshot Functions ---
async function saveAndviewResults() {
    const container = document.getElementById('object-list-container');
    const statusEl = document.getElementById('export-status');
    
    if (!container) return;
    
    // Show loading
    container.innerHTML = '<div class="col-span-full flex justify-center py-4"><div class="loader"></div></div>';
    
    try {
        const response = await postData("/api/snapshot/save", {});
        console.log("Snapshot response:", response);
        
        if (response && response.status === 'success' && response.objects.length > 0) {
            container.innerHTML = ''; // Clear loader
            
            response.objects.forEach(obj => {
                const item = document.createElement('div');
                item.className = 'bg-white p-2 rounded border border-gray-200 shadow-sm flex flex-col items-center gap-1 hover:border-primary transition-colors group';
                item.innerHTML = `
                    <div class="w-full aspect-square bg-[url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAHElEQVQYlWQQyQ0AAAyC7L/03YIqF8iJg3h5Dww0HwN7L8TjFAAAAABJRU5ErkJggg==')] bg-repeat rounded overflow-hidden flex items-center justify-center relative">
                         <img src="${obj.thumbnail}" class="max-w-full max-h-full object-contain z-10">
                         <div class="absolute top-1 left-1 bg-black/50 text-white text-[10px] px-1 rounded font-mono">#${obj.id}</div>
                    </div>
                    <span class="text-[10px] text-gray-500 font-medium truncate w-full text-center">object_${obj.id}.png</span>
                `;
                container.appendChild(item);
            });
            
            if (statusEl) {
                statusEl.classList.remove('hidden');
                setTimeout(() => statusEl.classList.add('hidden'), 5000);
            }
            
        } else {
            container.innerHTML = '<div class="col-span-full text-center text-sm text-gray-500 py-4">No detected objects found to save.<br><span class="text-xs text-gray-400">Run detection first.</span></div>';
        }
        
    } catch (error) {
        console.error("Error saving snapshot:", error);
        container.innerHTML = `<div class="col-span-full text-center text-xs text-red-500 py-2">Error: ${error.message}</div>`;
    }
}

// Add global click listener for modal closing and initialize app
document.addEventListener('DOMContentLoaded', () => {
    // 1. Modal Listeners
    const helpModal = document.getElementById('help-modal');
    if (helpModal) {
        helpModal.addEventListener('click', (e) => {
            if (e.target.id === 'help-modal') closeHelpModal();
        });
    }

    const inputHelpModal = document.getElementById('input-help-modal');
    if (inputHelpModal) {
        inputHelpModal.addEventListener('click', (e) => {
            if (e.target.id === 'input-help-modal') closeInputHelpModal();
        });
    }

    // 2. Initialize Components
    console.log("Initializing App Components...");
    setupSmartPromptDropdown();
    initializeInputModeSwitching();
    setupModelConfigListeners();
    setupSummaryListeners();
});
