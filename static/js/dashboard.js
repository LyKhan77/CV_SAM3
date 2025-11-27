// State management
let currentImageFile = null;
let isProcessing = false;

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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateDateTime();
    setInterval(updateDateTime, 1000);
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
}

async function handleRunAnalysis() {
    // Validation
    if (!currentImageFile) {
        alert('Please upload an image first');
        return;
    }

    const prompt = promptInput.value.trim();
    if (!prompt) {
        alert('Please enter an object name to detect');
        promptInput.focus();
        return;
    }

    if (isProcessing) {
        return;
    }

    isProcessing = true;
    setLoadingState(true);

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('prompt', prompt);

        // Send request
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Update image with processed result
            previewImage.src = result.data.result_image_url;

            // Update count display
            const detectedCount = result.data.detected_count;
            countDisplay.textContent = detectedCount;

            // Update progress bar and status
            updateProgressBar(detectedCount);
            updateStatusDisplay();

            // Show success message
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
    const detectedCount = parseInt(countDisplay.textContent.replace('{Count}', '0')) || 0;
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

    // Update progress bar if count exists
    if (!countDisplay.textContent.includes('{Count}')) {
        updateProgressBar(detectedCount);
    }
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
