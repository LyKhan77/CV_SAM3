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
    lastProcessStatus: "Ready"
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

// --- Modal Functions ---
function openHelpModal() {
    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.classList.remove('hidden');
        // Small timeout to allow display:block to apply before opacity transition
        setTimeout(() => modal.classList.remove('opacity-0'), 10);
    }
}

function closeHelpModal() {
    const modal = document.getElementById('help-modal');
    if (modal) {
        modal.classList.add('opacity-0');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }
}

function openInputHelpModal() {
    const modal = document.getElementById('input-help-modal');
    const content = document.getElementById('input-help-modal-content');
    if (modal) {
        modal.classList.remove('hidden');
        setTimeout(() => {
            modal.classList.remove('opacity-0');
            content.classList.remove('scale-95');
            content.classList.add('scale-100');
        }, 10);
    }
}

function closeInputHelpModal() {
    const modal = document.getElementById('input-help-modal');
    const content = document.getElementById('input-help-modal-content');
    if (modal) {
        modal.classList.add('opacity-0');
        content.classList.remove('scale-100');
        content.classList.add('scale-95');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }
}

function openModelHelpModal() {
    const modal = document.getElementById('model-help-modal');
    const content = document.getElementById('model-help-modal-content');
    if (modal) {
        modal.classList.remove('hidden');
        setTimeout(() => {
            modal.classList.remove('opacity-0');
            content.classList.remove('scale-95');
            content.classList.add('scale-100');
        }, 10);
    }
}

function closeModelHelpModal() {
    const modal = document.getElementById('model-help-modal');
    const content = document.getElementById('model-help-modal-content');
    if (modal) {
        modal.classList.add('opacity-0');
        content.classList.remove('scale-100');
        content.classList.add('scale-95');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }
}

// --- Toast Functions ---
let toastTimeout;
function showToast(message, type = 'warning') {
    const toast = document.getElementById('toast-notification');
    const border = document.getElementById('toast-border');
    const icon = document.getElementById('toast-icon');
    const title = document.getElementById('toast-title');
    const msgEl = document.getElementById('toast-message');

    if (toast && border && icon && title && msgEl) {
        clearTimeout(toastTimeout);

        // Configure Style based on Type
        if (type === 'success') {
            border.className = "bg-white border-l-4 border-green-500 shadow-xl rounded-lg p-4 flex items-start gap-3 max-w-sm";
            icon.className = "text-green-500 mt-0.5";
            icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            title.textContent = "Success";
        } else {
            // Default Warning
            border.className = "bg-white border-l-4 border-yellow-500 shadow-xl rounded-lg p-4 flex items-start gap-3 max-w-sm";
            icon.className = "text-yellow-500 mt-0.5";
            icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            title.textContent = "Notification";
        }

        // Set Content
        if (message) msgEl.textContent = message;

        // Show
        toast.classList.remove('translate-y-20', 'opacity-0');
        // Auto hide after 5 seconds
        toastTimeout = setTimeout(hideToast, 5000);
    }
}

function hideToast() {
    const toast = document.getElementById('toast-notification');
    if (toast) {
        toast.classList.add('translate-y-20', 'opacity-0');
    }
}


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

// Apply Quick Preset (Quality/Balanced/Speed)
async function applyPreset(preset) {
    try {
        console.log(`Applying preset: ${preset}`);
        const response = await postData('/api/config/preset', { preset: preset });

        if (response && response.status === 'success') {
            // Update UI sliders to reflect new values
            const config = response.config;
            if (config.confidence_threshold !== undefined) {
                const confidencePercent = Math.round(config.confidence_threshold * 100);
                document.getElementById('confidence-slider').value = confidencePercent;
                document.getElementById('confidence-value').textContent = `${confidencePercent}%`;
            }
            if (config.mask_threshold !== undefined) {
                const maskPercent = Math.round(config.mask_threshold * 100);
                document.getElementById('mask-slider').value = maskPercent;
                document.getElementById('mask-value').textContent = `${maskPercent}%`;
            }

            showToast(`✅ Applied ${preset.toUpperCase()} preset`, 'success');
            console.log(`Preset applied: ${JSON.stringify(config)}`);
        } else {
            showToast(`❌ Failed to apply preset: ${response?.message || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Error applying preset:', error);
        showToast(`❌ Error applying preset: ${error.message}`, 'error');
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
    const btn = document.getElementById('clear-mask-btn');
    const originalContent = btn ? btn.innerHTML : '';

    try {
        // UI Loading State
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<div class="loader w-4 h-4 border-red-500 border-t-transparent"></div> Clearing...';
        }

        await postData("/api/config/clear-mask", {});
        
        // Clear Visuals
        const descStatusEl = document.getElementById('desc-status');
        const statusDot = document.getElementById('status-dot');
        
        if (descStatusEl) descStatusEl.textContent = "Ready";
        if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-gray-400";
        
        // Unlock sliders
        document.getElementById('confidence-slider').disabled = false;
        document.getElementById('mask-slider').disabled = false;

        // Success Toast
        showToast("Mask cleared successfully.", "success");

    } catch (e) {
        console.error("Clear mask failed:", e);
        showToast("Failed to clear mask.", "warning");
    } finally {
        // Restore Button
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalContent;
        }
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

    // Video Performance Tuning Listeners
    const intervalSlider = document.getElementById('interval-slider');
    const intervalValue = document.getElementById('interval-value');

    if (intervalSlider && intervalValue) {
        intervalSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            intervalValue.textContent = value === '1' ? 'Every frame (Real-time)' : `Every ${value} frames`;
        });
        intervalSlider.addEventListener('change', async (e) => {
            const interval = parseInt(e.target.value);
            try {
                const response = await postData('/api/config/video/interval', { interval });
                if (response.status === 'success') {
                    console.log(`Processing interval updated to: ${interval}`);
                }
            } catch (error) {
                console.error('Error updating interval:', error);
            }
        });
    }
}

// --- 6.1 Video Performance Functions ---
async function setResolution(resolution) {
    try {
        const response = await postData('/api/config/video/resolution', { resolution });

        if (response.status === 'success') {
            // Update button states
            ['res-1024', 'res-768', 'res-512'].forEach(id => {
                const btn = document.getElementById(id);
                if (btn) {
                    if (id === `res-${resolution}`) {
                        btn.className = 'flex-1 px-2 py-1.5 text-xs font-medium text-white bg-primary rounded transition-colors';
                    } else {
                        btn.className = 'flex-1 px-2 py-1.5 text-xs font-medium text-gray-600 bg-gray-200 hover:bg-gray-300 rounded transition-colors';
                    }
                }
            });

            showToast(`Resolution set to ${resolution}px`, 'success');
            console.log(`Input resolution updated to: ${resolution}px`);
        }
    } catch (error) {
        console.error('Error updating resolution:', error);
        showToast('Failed to update resolution', 'error');
    }
}

// --- 6.2 Summary Panel Listeners ---
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
    const { video_frame, analytics } = data;

    // Handle Batch Processing Progress (Video Mode specific)
    if (data.status === "batch_processing" && analytics.batch_progress) {
        const progressContainer = document.getElementById('batch-progress-container');
        const progressBar = document.getElementById('batch-progress-bar');
        const progressText = document.getElementById('batch-progress-text');
        const playbackControls = document.getElementById('video-playback-controls');

        if (progressContainer) {
            progressContainer.classList.remove('hidden');
        }
        if (playbackControls) {
            playbackControls.classList.add('hidden'); // Hide playback controls during processing
        }

        const { current, total, percent } = analytics.batch_progress;

        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = `Frame ${current} / ${total} (${percent}%)`;
        }

        // Lock Run button
        const runBtn = document.getElementById('run-segmentation-btn');
        if (runBtn) {
            runBtn.disabled = true;
            runBtn.innerHTML = '<div class="loader w-4 h-4 border-white border-t-transparent"></div> Processing...';
        }

        // Lock all input controls
        const promptInput = document.getElementById('prompt-input');
        const confSlider = document.getElementById('confidence-slider');
        const maskSlider = document.getElementById('mask-slider');
        const intervalSlider = document.getElementById('interval-slider');

        if (promptInput) promptInput.disabled = true;
        if (confSlider) confSlider.disabled = true;
        if (maskSlider) maskSlider.disabled = true;
        if (intervalSlider) intervalSlider.disabled = true;

        // Lock mode switching
        document.querySelectorAll('input[name="input-mode"]').forEach(radio => {
            radio.disabled = true;
        });

        // Update status
        const statusEl = document.getElementById('desc-status');
        const statusDot = document.getElementById('status-dot');
        if (statusEl) statusEl.textContent = "Processing...";
        if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-yellow-400 animate-pulse";

        return; // Skip rest of update logic during batch processing
    }

    // Handle Video Mode UI states
    const progressContainer = document.getElementById('batch-progress-container');
    const playbackControls = document.getElementById('video-playback-controls');
    const currentMode = document.querySelector('input[name="input-mode"]:checked')?.value;

    if (currentMode === 'video') {
        // Show playback controls when Ready (waiting for batch) or Done (batch complete)
        if (playbackControls && (analytics.process_status === "Ready" || analytics.process_status === "Done")) {
            playbackControls.classList.remove('hidden');
        }

        // Hide progress bar when not processing
        if (progressContainer && analytics.process_status !== "Processing...") {
            progressContainer.classList.add('hidden');
        }

        // Unlock controls when Done or Ready
        if (analytics.process_status === "Done" || analytics.process_status === "Ready") {
            const promptInput = document.getElementById('prompt-input');
            const confSlider = document.getElementById('confidence-slider');
            const maskSlider = document.getElementById('mask-slider');
            const intervalSlider = document.getElementById('interval-slider');

            if (promptInput) promptInput.disabled = false;
            if (confSlider) confSlider.disabled = false;
            if (maskSlider) maskSlider.disabled = false;
            if (intervalSlider) intervalSlider.disabled = false;

            // Unlock mode switching
            document.querySelectorAll('input[name="input-mode"]').forEach(radio => {
                radio.disabled = false;
            });
        }
    }

    // 1. Update Video Feed
    const videoFeed = document.getElementById('mock-video-feed');
    if (videoFeed && video_frame) {
        videoFeed.src = video_frame;
        videoFeed.classList.remove('hidden');

        // If in RTSP/Video mode, ensure placeholder is hidden
        if (analytics.input_mode !== 'image') {
             document.getElementById('stream-placeholder').classList.add('hidden');
        }
    }

    // 2. Update Analytics Text
    const countEl = document.getElementById('detected-count');
    const statusEl = document.getElementById('desc-status');
    const statusDot = document.getElementById('status-dot');
    const descPromptEl = document.getElementById('desc-prompt');
    const runBtn = document.getElementById('run-segmentation-btn');

    if (countEl) animateValue(countEl, parseInt(countEl.textContent), analytics.count, 500);

    // Update Summary Panel (only for Image and RTSP modes, skip for Video mode)

    if (currentMode !== 'video') {
        // Update Progress Bar, Legend, and Result Badge
        const progressBar = document.getElementById('progress-bar');
        const progressLegend = document.getElementById('progress-legend');
        const resultBadge = document.getElementById('status-badge');

        if (progressBar && progressLegend) {
            const maxLimit = analytics.max_limit || 100;
            const percentage = Math.min((analytics.count / maxLimit) * 100, 100);

            progressBar.style.width = `${percentage}%`;
            // Color based on status
            if (analytics.status === "Approved") {
                 progressBar.className = "h-full rounded-full transition-all duration-500 ease-out bg-green-500";
            } else {
                 progressBar.className = "h-full rounded-full transition-all duration-500 ease-out bg-blue-500";
            }

            progressLegend.textContent = `${analytics.count}/${maxLimit}`;
        }

        // Update Result Badge
        if (resultBadge) {
            resultBadge.textContent = analytics.status;
            if (analytics.status === "Approved") {
                 resultBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-colors bg-green-200 text-green-800";
            } else {
                 resultBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-colors bg-blue-200 text-blue-800";
            }
        }
    }

    // Update Status Text & Dot
    if (statusEl) statusEl.textContent = analytics.process_status || "Ready";
    
    if (statusDot) {
        if (analytics.process_status === "Processing") {
            statusDot.className = "w-2 h-2 rounded-full bg-yellow-400 animate-pulse";
        } else if (analytics.status === "Approved") {
            statusDot.className = "w-2 h-2 rounded-full bg-green-500";
        } else {
            statusDot.className = "w-2 h-2 rounded-full bg-blue-500"; // Waiting/Ready default
        }
    }

    // Update Prompt Description
    if (descPromptEl) {
        descPromptEl.textContent = analytics.detected_object || "-";
    }
    
    // 3. Button State & Sliders
    // Unlock button if processing is done
    if ((analytics.process_status === "Done" || analytics.process_status === "Ready") && runBtn && runBtn.disabled) {
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Segmentation';
        
        // Unlock sliders (unless we want them locked while results are shown? No, usually unlock)
        document.getElementById('confidence-slider').disabled = false;
        document.getElementById('mask-slider').disabled = false;
    }
    
    // 4. Toast Notification Logic (Backend Driven)
    // Backend sends a warning message ONE TIME when it detects 0 results.
    if (analytics.warning) {
        showToast(analytics.warning, 'warning');
    }
    // frontendState.lastProcessStatus check is no longer needed for toasts but kept if used elsewhere
    frontendState.lastProcessStatus = analytics.process_status || "Ready";

    // 5. Video Specific Updates
    if (analytics.input_mode === 'video') {
         const videoSeek = document.getElementById('video-seek');
         if (videoSeek && !document.activeElement.isEqualNode(videoSeek)) {
             updateVideoProgress(analytics.video_current_frame, analytics.video_total_frames);
         }

         // Update FPS display
         const videoFpsDisplay = document.getElementById('video-fps');
         if (videoFpsDisplay && analytics.video_fps) {
             videoFpsDisplay.textContent = `${analytics.video_fps.toFixed(1)} FPS`;
         }

         // Sync play/pause button state
         const playPauseIcon = document.getElementById('play-pause-icon');
         const playPauseText = document.getElementById('play-pause-text');
         if (analytics.video_playing !== undefined) {
             if (playPauseIcon) {
                 playPauseIcon.className = analytics.video_playing ? 'fa-solid fa-pause' : 'fa-solid fa-play';
             }
             if (playPauseText) {
                 playPauseText.textContent = analytics.video_playing ? 'Pause' : 'Play';
             }
         }
    }

    // 6. Sound Trigger
    if (analytics.trigger_sound) {
        triggerNotification();
    }

    frontendState.currentCount = analytics.count;
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

    // Hide video-specific controls when leaving video mode
    const videoPlaybackControls = document.getElementById('video-playback-controls');
    const videoFileInfo = document.getElementById('video-file-info');
    if (videoPlaybackControls) videoPlaybackControls.classList.add('hidden');
    if (videoFileInfo) videoFileInfo.classList.add('hidden');

    // Clear prompts UI when switching modes
    const promptInput = document.getElementById('prompt-input');
    const descPrompt = document.getElementById('desc-prompt');
    if (promptInput) promptInput.value = '';
    if (descPrompt) descPrompt.textContent = 'Click "Run Mask" to start';

    // Reset state for Video mode - clear detected objects message
    if (mode === 'video') {
        const objectListContainer = document.getElementById('object-list-container');
        if (objectListContainer) {
            objectListContainer.innerHTML = '<p class="col-span-full text-center text-sm text-gray-400">Export not available in Video mode</p>';
        }
    }

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
        } else {
             // File already loaded, show feed and controls
             if (overlay) overlay.classList.add('hidden');
             // Show video controls if video is loaded
             if (videoFileInfo) videoFileInfo.classList.remove('hidden');
             if (videoPlaybackControls) videoPlaybackControls.classList.remove('hidden');
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

    // 4. Update mode-specific panel visibility
    updateModePanelVisibility(mode);

    // 5. Notify Backend
    try {
        await postData('/api/config/input-mode', { mode: mode });
    } catch (error) {
        console.error('Error switching input mode:', error);
    }
}

function updateModePanelVisibility(mode) {
    // Hide all mode-specific panels first
    document.querySelectorAll('[data-mode]').forEach(panel => {
        panel.classList.add('hidden');
    });

    // Show panels for current mode
    document.querySelectorAll(`[data-mode*="${mode}"]`).forEach(panel => {
        panel.classList.remove('hidden');
    });

    console.log(`Updated panel visibility for mode: ${mode}`);
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

// Cancel batch processing
async function cancelBatchProcessing() {
    try {
        await postData('/api/config/clear-mask', {});
        showToast('Batch processing cancelled', 'warning');

        // Reset UI
        const progressContainer = document.getElementById('batch-progress-container');
        if (progressContainer) {
            progressContainer.classList.add('hidden');
        }

        const runBtn = document.getElementById('run-segmentation-btn');
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Segmentation';
        }

        // Unlock controls
        const promptInput = document.getElementById('prompt-input');
        const confSlider = document.getElementById('confidence-slider');
        const maskSlider = document.getElementById('mask-slider');
        const intervalSlider = document.getElementById('interval-slider');

        if (promptInput) promptInput.disabled = false;
        if (confSlider) confSlider.disabled = false;
        if (maskSlider) maskSlider.disabled = false;
        if (intervalSlider) intervalSlider.disabled = false;

        // Unlock mode switching
        document.querySelectorAll('input[name="input-mode"]').forEach(radio => {
            radio.disabled = false;
        });
    } catch (error) {
        console.error('Error cancelling batch processing:', error);
    }
}

async function clearVideo() {
    try {
        const response = await postData('/api/config/video/clear', {});

        if (response && response.status === 'success') {
            // Hide video controls if they exist
            const videoPlaybackControls = document.getElementById('video-playback-controls');
            if (videoPlaybackControls) {
                videoPlaybackControls.classList.add('hidden');
            }

            // Clear video file input
            const videoFileInput = document.getElementById('video-file');
            if (videoFileInput) {
                videoFileInput.value = '';
            }

            // Hide video file info
            const videoFileInfo = document.getElementById('video-file-info');
            if (videoFileInfo) {
                videoFileInfo.classList.add('hidden');
            }

            // Hide video feed and show placeholder
            const mockVideoFeed = document.getElementById('mock-video-feed');
            const streamPlaceholder = document.getElementById('stream-placeholder');

            if (mockVideoFeed) {
                mockVideoFeed.classList.add('hidden');
                // Clear the image source to remove residual frame
                const videoImage = mockVideoFeed.querySelector('img');
                if (videoImage) {
                    videoImage.src = '';
                }
            }

            if (streamPlaceholder) {
                streamPlaceholder.classList.remove('hidden');
                // Reset placeholder to default video mode message
                streamPlaceholder.innerHTML = `
                    <p class="text-gray-400 text-lg mb-2">Drop Video File Here</p>
                    <p class="text-gray-400 text-sm mb-4">or</p>
                    <button onclick="document.getElementById('video-file').click()" class="bg-primary hover:bg-blue-800 text-white font-bold px-6 py-2 rounded-lg text-sm transition-colors shadow-md pointer-events-auto relative z-50">
                        <i class="fa-solid fa-folder-open mr-2"></i>
                        Browse Files
                    </button>
                `;
            }

            // Clear Object List
            const listContainer = document.getElementById('object-list-container');
            if (listContainer) {
                listContainer.innerHTML = '<p class="col-span-full text-center text-sm">No objects saved yet.</p>';
            }

            showToast('Video cleared', 'success');
        }
    } catch (error) {
        console.error('Error clearing video:', error);
        showToast('Failed to clear video', 'error');
    }
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
            
            // Clear Prompt UI (New)
            const promptInput = document.getElementById('prompt-input');
            const descPrompt = document.getElementById('desc-prompt');
            if (promptInput) promptInput.value = '';
            if (descPrompt) descPrompt.textContent = '-';

            // Reset Analytics UI
            const countEl = document.getElementById('detected-count');
            const progressBar = document.getElementById('progress-bar');
            const progressLegend = document.getElementById('progress-legend');
            const resultBadge = document.getElementById('status-badge');
            const statusEl = document.getElementById('desc-status');
            const statusDot = document.getElementById('status-dot');

            if (countEl) countEl.textContent = "0";
            if (progressBar) progressBar.style.width = "0%";
            if (progressLegend) progressLegend.textContent = "0/100";
            
            if (resultBadge) {
                resultBadge.textContent = "Waiting";
                resultBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-colors bg-blue-200 text-blue-800";
            }
            
            if (statusEl) statusEl.textContent = "Ready";
            if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-gray-400";

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
            
            showToast("File removed.", "success");
            
            switchInputMode('image');
        }
    } catch (error) {
        console.error('Error clearing uploaded image:', error);
    }
}

function updateVideoProgress(currentFrame, totalFrames) {
    const videoSeek = document.getElementById('video-seek');
    const frameCounter = document.getElementById('frame-counter');

    if (videoSeek && totalFrames > 0) {
        videoSeek.max = totalFrames;
        videoSeek.value = currentFrame || 0;
    }

    if (frameCounter) {
        frameCounter.textContent = `${currentFrame || 0} / ${totalFrames || 0}`;
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

        const response = await postData('/api/config/video/toggle', {});

        if (response && response.status === 'success') {
            const playPauseBtn = document.getElementById('play-pause-btn');
            const playPauseIcon = document.getElementById('play-pause-icon');
            const playPauseText = document.getElementById('play-pause-text');

            // Update UI based on current state
            if (response.video_playing) {
                if (playPauseIcon) playPauseIcon.className = 'fa-solid fa-pause';
                if (playPauseText) playPauseText.textContent = 'Pause';
                console.log("Video playing");
            } else {
                if (playPauseIcon) playPauseIcon.className = 'fa-solid fa-play';
                if (playPauseText) playPauseText.textContent = 'Play';
                console.log("Video paused");
            }

            showToast(`Video ${response.video_playing ? 'playing' : 'paused'}`, 'info');
        }
    } catch (error) {
        console.error('Error toggling video playback:', error);
        showToast('Failed to toggle playback', 'error');
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
