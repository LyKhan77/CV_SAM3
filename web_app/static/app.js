// --- 0. Constants and Configuration ---
const API_BASE_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/monitor";

// --- 1. Utilities (Time) ---
function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    const dateString = now.toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'short', year: 'numeric' });
    
    document.getElementById('clock-time').textContent = timeString;
    document.getElementById('clock-date').textContent = dateString;
}
setInterval(updateTime, 1000);
updateTime();

// --- 2. State Management ---
// The primary state is now managed by the backend.
// This frontend state is used to track the "current" value for animation purposes.
let frontendState = {
    currentCount: 0,
};

// --- 3. WebSocket Connection ---
const socket = new WebSocket(WS_URL);

socket.onopen = (event) => {
    console.log("WebSocket connection established.");
    socket.send("Hello Server!"); // Send a message to confirm connection
};

socket.onmessage = (event) => {
    // The backend sends data as a string, so we need to parse it.
    // Using a try-catch block to handle potential JSON parsing errors.
    try {
        // The mock data from the backend is a string representation of a dictionary,
        // which needs to be cleaned up (replace single quotes with double quotes) to be valid JSON.
        const cleanedDataString = event.data.replace(/'/g, '"');
        const data = JSON.parse(cleanedDataString);
        console.log("Received data:", data);
        updateDashboard(data);
    } catch (e) {
        console.error("Failed to parse WebSocket message:", e, "Raw data:", event.data);
    }
};

socket.onclose = (event) => {
    console.log("WebSocket connection closed.");
    // Optionally, handle reconnection logic here.
};

socket.onerror = (error) => {
    console.error("WebSocket error:", error);
};


// --- 4. API Call Functions ---

async function postData(endpoint, body) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (e) {
        console.error(`Failed to post to ${endpoint}:`, e);
        alert(`Error communicating with the backend. Is it running?`);
    }
}

// --- 5. UI Event Handlers ---

// A. Streaming Panel Logic
function activateStream() {
    const url = document.getElementById('rtsp-input').value;
    if (url.trim() === "") {
        alert("Please enter a valid RTSP URL");
        return;
    }
    
    const placeholder = document.getElementById('stream-placeholder');
    placeholder.innerHTML = '<div class="loader"></div><p class="mt-2 text-sm text-gray-500">Connecting to RTSP...</p>';
    
    postData("/api/config/stream", { url: url }).then(data => {
        console.log("Stream activation response:", data);
        // The backend will now start sending video data via WebSocket.
        // We just need to show the video feed area.
        const liveIndicator = document.getElementById('live-indicator');
        const videoFeed = document.getElementById('mock-video-feed');
        
        placeholder.classList.add('hidden');
        liveIndicator.classList.remove('hidden');
        videoFeed.classList.remove('hidden');
    });
}

// B. Prompt & Description Logic
function processPrompt() {
    const inputVal = document.getElementById('prompt-input').value;
    if (inputVal.trim() === "") return;

    const descEl = document.getElementById('object-description');
    descEl.innerHTML = '<div class="loader"></div>';
    
    postData("/api/config/prompt", { object_name: inputVal }).then(data => {
        console.log("Prompt set response:", data);
        // The backend will confirm the prompt in the next WebSocket message.
    });
}

// C. Limit Logic
function updateLimit() {
    const limit = parseInt(document.getElementById('max-limit').value) || 100;
    postData("/api/config/limit", { value: limit }).then(data => {
        console.log("Limit set response:", data);
    });
}

// D. Sound Toggle Logic
function updateSoundSetting() {
    const enabled = document.getElementById('sound-toggle').checked;
    postData("/api/config/sound", { enabled: enabled }).then(data => {
        console.log("Sound setting response:", data);
    });
}


// --- 6. DOM Element Event Listeners ---
document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('max-limit').addEventListener('change', updateLimit);
    document.getElementById('sound-toggle').addEventListener('change', updateSoundSetting);
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
    
    // Animate the count
    animateValue(countEl, frontendState.currentCount, analytics.count, 500);
    frontendState.currentCount = analytics.count;

    // Update video feed (assuming base64 for now)
    if(data.video_frame && data.video_frame.startsWith('data:image')) {
       videoFeed.src = data.video_frame;
    }

    // Update description
    if (analytics.detected_object && analytics.detected_object !== "N/A") {
        descEl.innerHTML = `Detecting: <span class="font-bold text-primary">${analytics.detected_object}</span>`;
    }
    
    // Update progress bar
    let percentage = (analytics.count / analytics.max_limit) * 100;
    if (percentage > 100) percentage = 100;
    progressBar.style.width = `${percentage}%`;

    // Update legend
    progressLegend.textContent = `${analytics.count}/${analytics.max_limit}`;
    document.getElementById('max-limit').value = analytics.max_limit;

    // Update status badge
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
    } else { // Over limit / Red
        statusBadge.className = "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-red-200 text-red-800 transition-colors duration-300 animate-pulse";
        progressBar.classList.remove('bg-primary');
        progressBar.classList.add('bg-red-600');
    }

    // Trigger sound
    if (analytics.trigger_sound) {
        triggerNotification();
    }
}

function triggerNotification() {
    const audio = document.getElementById('notification-sound');
    audio.currentTime = 0;
    audio.play().catch(e => console.log("Audio play failed (interaction required first):", e));
}

// Helper: Animate Numbers - Modified to not call updateStatus at the end
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
