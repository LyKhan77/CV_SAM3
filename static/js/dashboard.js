// State management
let currentImageFile = null;
let isProcessing = false;

// Instant segmentation state (Select Object mode)
let segmentedObjects = [];
let currentObjectId = 0;
let isSegmenting = false;

// Toast notification system
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg shadow-lg text-white z-50 transition-opacity duration-300`;

    const colors = {
        'info': 'bg-blue-500',
        'warning': 'bg-orange-500',
        'error': 'bg-red-500',
        'success': 'bg-green-500'
    };

    toast.classList.add(colors[type] || colors.info);
    toast.textContent = message;

    document.body.appendChild(toast);

    // Fade out and remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const promptInput = document.getElementById('promptInput');
const runButton = document.getElementById('runButton');
const maxLimitInput = document.getElementById('maxLimitInput');
const statusCard = document.getElementById('statusCard');
const countDisplay = document.getElementById('countDisplay');
const progressBar = document.getElementById('progressBar');
const progressLabel = document.getElementById('progressLabel');

// Image card elements
const imageCard = document.getElementById('imageCard');
const imageThumbnail = document.getElementById('imageThumbnail');
const imageFileName = document.getElementById('imageFileName');
const removeImageBtn = document.getElementById('removeImageBtn');

// Modal elements
const promptModal = document.getElementById('promptModal');
const howToPromptBtn = document.getElementById('howToPromptBtn');
const closeModalBtn = document.getElementById('closeModalBtn');

// Model Configuration elements
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const selectObjectToggle = document.getElementById('selectObjectToggle');
const modeDescription = document.getElementById('modeDescription');

// Click mode state
let detectionMode = 'text'; // 'text' or 'click'
let clickedPoints = [];
let clickMarkers = [];

// Load model information from backend
async function loadModelInfo() {
    try {
        const response = await fetch('/model/info');
        const result = await response.json();

        if (result.success) {
            const modelNameEl = document.getElementById('modelName');
            modelNameEl.textContent = result.model_name;
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateDateTime();
    setInterval(updateDateTime, 1000);
    loadModelInfo();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-[#003473]');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-[#003473]');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-[#003473]');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Run button click
    runButton.addEventListener('click', handleRunAnalysis);

    // Max limit change - update status
    maxLimitInput.addEventListener('input', updateStatusDisplay);

    // Allow Enter key in prompt input
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isProcessing) {
            handleRunAnalysis();
        }
    });

    // Remove image button
    removeImageBtn.addEventListener('click', handleRemoveImage);

    // Modal handlers
    howToPromptBtn.addEventListener('click', () => {
        promptModal.classList.remove('hidden');
    });

    closeModalBtn.addEventListener('click', () => {
        promptModal.classList.add('hidden');
    });

    // Close modal on backdrop click
    promptModal.addEventListener('click', (e) => {
        if (e.target === promptModal) {
            promptModal.classList.add('hidden');
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !promptModal.classList.contains('hidden')) {
            promptModal.classList.add('hidden');
        }
    });

    // Model Configuration handlers
    confidenceSlider.addEventListener('input', (e) => {
        confidenceValue.textContent = e.target.value;
    });

    selectObjectToggle.addEventListener('change', (e) => {
        detectionMode = e.target.checked ? 'click' : 'text';

        if (detectionMode === 'click') {
            modeDescription.textContent = 'Select Object mode: Click objects to instantly segment and identify';
            enableClickMode();
            updateRunButtonForMode('click');
        } else {
            modeDescription.textContent = 'Text mode: Enter prompt to auto-detect objects';
            disableClickMode();
            updateRunButtonForMode('text');
        }
    });
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, JPEG, or PNG)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    currentImageFile = file;

    // Preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
        const imageData = e.target.result;

        // Update main preview
        previewImage.src = imageData;
        uploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');

        // Update image card (thumbnail + filename)
        imageThumbnail.src = imageData;
        imageFileName.textContent = file.name;
        imageCard.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function handleRemoveImage() {
    // Clear state
    currentImageFile = null;
    imageInput.value = '';

    // Hide image card and preview
    imageCard.classList.add('hidden');
    imagePreview.classList.add('hidden');

    // Show upload area again
    uploadArea.classList.remove('hidden');

    // Clear thumbnail and filename
    imageThumbnail.src = '';
    imageFileName.textContent = '';

    // Clear click markers if in click mode
    clearClickMarkers();
}

async function handleRunAnalysis() {
    // Validation
    if (!currentImageFile) {
        alert('Please upload an image first');
        return;
    }

    if (isProcessing) {
        return;
    }

    // Route to appropriate mode
    if (detectionMode === 'click') {
        if (clickedPoints.length === 0) {
            alert('Please click on objects in the image first');
            return;
        }
        await runClickDetection();
    } else {
        // Text mode
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter an object name to detect');
            promptInput.focus();
            return;
        }
        await runTextDetection(prompt);
    }
}

async function runTextDetection(prompt) {
    isProcessing = true;
    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('prompt', prompt);
        formData.append('mode', 'text');
        formData.append('confidence', confidenceSlider.value);

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            previewImage.src = result.data.result_image_url;
            countDisplay.textContent = result.data.detected_count;
            updateProgressBar(result.data.detected_count);
            updateStatusDisplay();
            console.log('Analysis complete:', result.message);
        } else {
            alert('Error: ' + result.error);
        }

    } catch (error) {
        console.error('Error during analysis:', error);
        alert('An error occurred while processing the image. Please try again.');
    } finally {
        isProcessing = false;
        setLoadingState(false);
    }
}

async function runClickDetection() {
    isProcessing = true;
    setLoadingState(true);

    try {
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('mode', 'click');
        formData.append('points', JSON.stringify(clickedPoints));
        formData.append('confidence', confidenceSlider.value);

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            previewImage.src = result.data.result_image_url;
            countDisplay.textContent = result.data.detected_count;
            updateProgressBar(result.data.detected_count);
            updateStatusDisplay();
            clearClickMarkers(); // Clear markers after processing
            console.log('Analysis complete:', result.message);
        } else {
            alert('Error: ' + result.error);
        }

    } catch (error) {
        console.error('Error during analysis:', error);
        alert('An error occurred while processing the image. Please try again.');
    } finally {
        isProcessing = false;
        setLoadingState(false);
    }
}

function setLoadingState(loading) {
    if (loading) {
        runButton.disabled = true;
        runButton.innerHTML = `
            <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
        `;
    } else {
        runButton.disabled = false;
        runButton.innerHTML = `
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
            </svg>
            Run
        `;
    }
}

function updateProgressBar(count) {
    const maxLimit = parseInt(maxLimitInput.value) || 0;
    const percentage = maxLimit > 0 ? Math.min((count / maxLimit) * 100, 100) : 0;

    progressBar.style.width = `${percentage}%`;
    progressLabel.textContent = `(${count})/(${maxLimit})`;
}

function updateStatusDisplay() {
    const detectedCount = parseInt(countDisplay.textContent) || 0;
    const maxLimit = parseInt(maxLimitInput.value) || 0;

    if (detectedCount >= maxLimit && maxLimit > 0) {
        // Approved state
        statusCard.className = 'p-3 rounded-lg text-center font-bold bg-green-50 border-2 border-green-200 text-green-700';
        statusCard.textContent = 'Approved';
    } else {
        // Waiting state
        statusCard.className = 'p-3 rounded-lg text-center font-bold bg-orange-50 border-2 border-orange-200 text-orange-700';
        statusCard.textContent = 'Waiting';
    }

    // Update progress bar
    updateProgressBar(detectedCount);
}

function updateDateTime() {
    const now = new Date();

    const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const dateOptions = { year: 'numeric', month: '2-digit', day: '2-digit' };

    const timeStr = now.toLocaleTimeString('en-US', timeOptions);
    const dateStr = now.toLocaleDateString('en-US', dateOptions);

    document.querySelectorAll('header .text-right div')[0].textContent = timeStr;
    document.querySelectorAll('header .text-right div')[1].textContent = dateStr;
}

// Click mode functions
function enableClickMode() {
    previewImage.style.cursor = 'crosshair';
    previewImage.addEventListener('click', handleImageClick);
}

function disableClickMode() {
    previewImage.style.cursor = 'default';
    previewImage.removeEventListener('click', handleImageClick);
    clearClickMarkers();
    clearAllSegments();
}

function updateRunButtonForMode(mode) {
    if (mode === 'click') {
        // Change Run button to Clear Segments button
        runButton.innerHTML = `
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
            Clear Segments
        `;
        runButton.onclick = (e) => {
            e.preventDefault();
            clearAllSegments();
        };
    } else {
        // Reset to Run button
        runButton.innerHTML = `
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
            </svg>
            Run
        `;
        runButton.onclick = (e) => {
            e.preventDefault();
            handleRunAnalysis();
        };
    }
}

async function handleImageClick(e) {
    if (isSegmenting) {
        return; // Prevent multiple simultaneous clicks
    }

    const rect = previewImage.getBoundingClientRect();
    const scaleX = previewImage.naturalWidth / rect.width;
    const scaleY = previewImage.naturalHeight / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;

    // Instant segmentation - process immediately
    await performInstantSegmentation({x, y, label: 1}, displayX, displayY);
}

function addClickMarker(x, y) {
    const marker = document.createElement('div');
    marker.className = 'absolute w-3 h-3 bg-red-500 rounded-full border-2 border-white';
    marker.style.left = `${x - 6}px`;
    marker.style.top = `${y - 6}px`;
    marker.style.pointerEvents = 'none';

    const container = previewImage.parentElement;
    container.style.position = 'relative';
    container.appendChild(marker);

    clickMarkers.push(marker);

    return marker;  // IMPORTANT: Return the marker element
}

function clearClickMarkers() {
    clickMarkers.forEach(marker => marker.remove());
    clickMarkers = [];
    clickedPoints = [];
}

// Instant segmentation functions (Select Object mode)
async function performInstantSegmentation(point, displayX, displayY) {
    if (!currentImageFile) {
        return;
    }

    isSegmenting = true;

    // Show loading cursor
    previewImage.style.cursor = 'wait';

    try {
        // Show temporary loading marker at click point
        const loadingMarker = addClickMarker(displayX, displayY);
        loadingMarker.classList.add('animate-pulse');

        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('x', point.x);
        formData.append('y', point.y);
        formData.append('confidence', confidenceSlider.value);

        const response = await fetch('/analyze/instant', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Remove loading marker
        loadingMarker.remove();
        clickMarkers = clickMarkers.filter(m => m !== loadingMarker);

        if (result.success) {
            const data = result.data;

            // Filter by confidence threshold
            const confidenceThreshold = parseFloat(confidenceSlider.value);
            if (data.confidence < confidenceThreshold) {
                const confidencePercent = (data.confidence * 100).toFixed(1);
                const thresholdPercent = (confidenceThreshold * 100).toFixed(1);
                const message = `Object rejected: ${data.object_label} (${confidencePercent}%) below threshold (${thresholdPercent}%)`;

                showToast(message, 'warning');
                console.log(message);
                return;  // Exit early - don't add to segmentedObjects
            }

            // Check for IoU overlap with existing objects
            const overlappingObject = findOverlappingObject(data.bbox);

            if (overlappingObject) {
                console.log(`Object overlaps with existing object #${overlappingObject.id} - skipping`);
                return;
            }

            // Add new segmented object
            const objectId = currentObjectId++;
            segmentedObjects.push({
                id: objectId,
                bbox: data.bbox,
                label: data.object_label,
                confidence: data.confidence,
                mask: data.mask,
                mask_shape: data.mask_shape,
                top_3: data.top_3,
                displayX: displayX,
                displayY: displayY
            });

            // Render overlay on canvas
            renderSegmentOverlay(objectId, data);

            // Update count display
            updateSegmentCount();

            console.log(`Object #${objectId}: ${data.object_label} (${(data.confidence * 100).toFixed(1)}%)`);
        } else {
            console.error('Instant segmentation failed:', result.error);
            alert('Segmentation failed: ' + result.error);
        }

    } catch (error) {
        console.error('Error during instant segmentation:', error);
        alert('An error occurred. Please try again.');
    } finally {
        isSegmenting = false;
        previewImage.style.cursor = 'crosshair';
    }
}

function findOverlappingObject(newBbox) {
    // Check IoU (Intersection over Union) with existing objects
    for (const obj of segmentedObjects) {
        const iou = calculateIoU(newBbox, obj.bbox);
        if (iou > 0.5) {
            // >50% overlap - consider as same object
            return obj;
        }
    }
    return null;
}

function calculateIoU(bbox1, bbox2) {
    // bbox format: [x1, y1, x2, y2]
    const [x1_1, y1_1, x2_1, y2_1] = bbox1;
    const [x1_2, y1_2, x2_2, y2_2] = bbox2;

    // Calculate intersection area
    const x_left = Math.max(x1_1, x1_2);
    const y_top = Math.max(y1_1, y1_2);
    const x_right = Math.min(x2_1, x2_2);
    const y_bottom = Math.min(y2_1, y2_2);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0; // No intersection
    }

    const intersection = (x_right - x_left) * (y_bottom - y_top);

    // Calculate union area
    const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    const union = area1 + area2 - intersection;

    return intersection / union;
}

function renderSegmentOverlay(objectId, data) {
    // Create canvas overlay
    const container = previewImage.parentElement;

    if (!container.style.position || container.style.position === 'static') {
        container.style.position = 'relative';
    }

    let canvas = container.querySelector('canvas.segment-overlay');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.className = 'segment-overlay absolute top-0 left-0 pointer-events-none';
        canvas.style.zIndex = '10';

        const rect = previewImage.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;

        container.appendChild(canvas);
    }

    const ctx = canvas.getContext('2d');
    const rect = previewImage.getBoundingClientRect();
    const scaleX = rect.width / previewImage.naturalWidth;
    const scaleY = rect.height / previewImage.naturalHeight;
    const color = getSegmentColor(objectId);

    // Draw filled mask with contour outline
    if (data.contour && data.contour.length > 0) {
        ctx.beginPath();

        const firstPoint = data.contour[0];
        ctx.moveTo(firstPoint[0] * scaleX, firstPoint[1] * scaleY);

        for (let i = 1; i < data.contour.length; i++) {
            const point = data.contour[i];
            ctx.lineTo(point[0] * scaleX, point[1] * scaleY);
        }

        ctx.closePath();

        // Fill with semi-transparent color
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`;
        ctx.fill();

        // Draw contour outline
        ctx.strokeStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
        ctx.lineWidth = 2;
        ctx.stroke();
    } else {
        // Fallback: Draw bounding box if contour not available
        const [x1, y1, x2, y2] = data.bbox;
        const canvasX1 = x1 * scaleX;
        const canvasY1 = y1 * scaleY;
        const canvasX2 = x2 * scaleX;
        const canvasY2 = y2 * scaleY;

        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`;
        ctx.fillRect(canvasX1, canvasY1, canvasX2 - canvasX1, canvasY2 - canvasY1);

        ctx.strokeStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(canvasX1, canvasY1, canvasX2 - canvasX1, canvasY2 - canvasY1);
    }

    // Draw label at bbox center
    const [x1, y1, x2, y2] = data.bbox;
    const centerX = ((x1 + x2) / 2) * scaleX;
    const centerY = ((y1 + y2) / 2) * scaleY;

    const label = `${data.object_label} (${(data.confidence * 100).toFixed(0)}%)`;
    ctx.font = 'bold 14px Arial';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;

    ctx.strokeText(label, centerX - ctx.measureText(label).width / 2, centerY);
    ctx.fillText(label, centerX - ctx.measureText(label).width / 2, centerY);
}

function getSegmentColor(index) {
    // Color palette for overlays
    const colors = [
        { r: 0, g: 52, b: 115 },   // Primary color
        { r: 255, g: 140, b: 0 },  // Orange
        { r: 0, g: 255, b: 0 },    // Green
        { r: 255, g: 0, b: 0 },    // Red
        { r: 255, g: 255, b: 0 },  // Yellow
        { r: 255, g: 0, b: 255 },  // Magenta
    ];
    return colors[index % colors.length];
}

function updateSegmentCount() {
    const count = segmentedObjects.length;
    countDisplay.textContent = count;
    updateProgressBar(count);
    updateStatusDisplay();
    updateDescription();  // Update description panel with object labels
}

function updateDescription() {
    // Find the description paragraph element
    const descriptionPanel = document.querySelector('.bg-white.rounded-xl.shadow-sm .bg-gray-50 p');

    if (!descriptionPanel) {
        console.error('Description panel not found');
        return;
    }

    if (segmentedObjects.length === 0) {
        descriptionPanel.textContent = '{Object Description}';
    } else if (segmentedObjects.length === 1) {
        descriptionPanel.textContent = `Detected: ${segmentedObjects[0].label}`;
    } else {
        // Multiple objects - show unique labels
        const uniqueLabels = [...new Set(segmentedObjects.map(obj => obj.label))];
        descriptionPanel.textContent = `Detected: ${uniqueLabels.join(', ')}`;
    }
}

function clearAllSegments() {
    // Remove canvas overlay
    const container = previewImage.parentElement;
    const canvas = container.querySelector('canvas.segment-overlay');
    if (canvas) {
        canvas.remove();
    }

    // Clear state
    segmentedObjects = [];
    currentObjectId = 0;

    // Reset count and description
    countDisplay.textContent = '0';
    updateProgressBar(0);
    updateStatusDisplay();
    updateDescription();  // Reset description to placeholder

    console.log('All segments cleared');
}
