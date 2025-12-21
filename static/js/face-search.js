/**
 * Face Search JavaScript
 * =======================
 * 
 * Handles camera capture, file upload, API communication,
 * and results display for the Find My Photos page.
 * 
 * Features:
 * - Camera capture (webcam/mobile front camera)
 * - File upload
 * - Paginated results with "See More"
 * - Lightbox for full-size photo viewing
 * - Photo selection and download
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE = '/api/face';
const POLL_INTERVAL = 1000; // 1 second
const PHOTOS_PER_PAGE = 15; // Number of thumbnails to show initially / per "See More"

// =============================================================================
// STATE
// =============================================================================

let currentTaskId = null;
let selectedPhotos = new Set();
let allPhotos = [];
let displayedCount = 0; // How many photos are currently displayed
let mediaStream = null;
let currentLightboxIndex = -1; // Currently viewing photo in lightbox
let previewImageUrl = null; // Store preview of uploaded/captured photo

// Store DOM elements (will be populated after DOM loads)
let elements = {};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🔍 Face Search JS loaded');

    // Get all DOM elements after DOM is ready
    elements = {
        // Upload section
        uploadSection: document.getElementById('upload-section'),
        cameraBtn: document.getElementById('camera-btn'),
        fileInput: document.getElementById('file-input'),
        threshold: document.getElementById('threshold'),

        // Camera view
        cameraView: document.getElementById('camera-view'),
        cameraPreview: document.getElementById('camera-preview'),
        captureBtn: document.getElementById('capture-btn'),
        cancelCameraBtn: document.getElementById('cancel-camera-btn'),

        // Loading
        loadingSection: document.getElementById('loading-section'),
        loadingMessage: document.getElementById('loading-message'),

        // Results
        resultsSection: document.getElementById('results-section'),
        resultCount: document.getElementById('result-count'),
        resultSubtitle: document.getElementById('result-subtitle'),
        photosGrid: document.getElementById('photos-grid'),
        selectAllBtn: document.getElementById('select-all-btn'),
        downloadSelectedBtn: document.getElementById('download-selected-btn'),
        shareToMemoriesBtn: document.getElementById('share-to-memories-btn'),
        selectedCount: document.getElementById('selected-count'),
        newSearchBtn: document.getElementById('new-search-btn'),
        seeMoreBtn: document.getElementById('see-more-btn'),
        showingCount: document.getElementById('showing-count'),

        // No results
        noResultsSection: document.getElementById('no-results-section'),
        tryAgainBtn: document.getElementById('try-again-btn'),

        // Error
        errorSection: document.getElementById('error-section'),
        errorMessage: document.getElementById('error-message'),
        errorRetryBtn: document.getElementById('error-retry-btn'),

        // Lightbox
        lightbox: document.getElementById('lightbox'),
        lightboxImage: document.getElementById('lightbox-image'),
        lightboxClose: document.getElementById('lightbox-close'),
        lightboxPrev: document.getElementById('lightbox-prev'),
        lightboxNext: document.getElementById('lightbox-next'),
        lightboxDownload: document.getElementById('lightbox-download'),
        lightboxCounter: document.getElementById('lightbox-counter'),
        lightboxInfo: document.getElementById('lightbox-info'),

        // Loading progress
        previewPhoto: document.getElementById('preview-photo'),
        loadingDetail: document.getElementById('loading-detail'),
        progressStep2: document.getElementById('progress-step-2'),
        progressStep3: document.getElementById('progress-step-3'),
        progressBar2: document.getElementById('progress-bar-2'),
    };

    // Debug: Log which elements were found
    console.log('Camera button found:', !!elements.cameraBtn);
    console.log('Camera view found:', !!elements.cameraView);
    console.log('Upload section found:', !!elements.uploadSection);

    // Attach event listeners
    if (elements.cameraBtn) {
        elements.cameraBtn.addEventListener('click', startCamera);
        console.log('✓ Camera button click handler attached');
    } else {
        console.error('❌ Camera button not found!');
    }

    if (elements.cancelCameraBtn) {
        elements.cancelCameraBtn.addEventListener('click', stopCamera);
    }

    if (elements.captureBtn) {
        elements.captureBtn.addEventListener('click', capturePhoto);
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileUpload);
    }

    // Results action buttons
    if (elements.selectAllBtn) {
        elements.selectAllBtn.addEventListener('click', toggleSelectAll);
    }
    if (elements.downloadSelectedBtn) {
        elements.downloadSelectedBtn.addEventListener('click', downloadSelected);
    }
    if (elements.shareToMemoriesBtn) {
        elements.shareToMemoriesBtn.addEventListener('click', openShareModal);
    }
    if (elements.newSearchBtn) {
        elements.newSearchBtn.addEventListener('click', resetSearch);
    }
    if (elements.tryAgainBtn) {
        elements.tryAgainBtn.addEventListener('click', resetSearch);
    }
    if (elements.errorRetryBtn) {
        elements.errorRetryBtn.addEventListener('click', resetSearch);
    }
    if (elements.seeMoreBtn) {
        elements.seeMoreBtn.addEventListener('click', showMorePhotos);
    }

    // Lightbox controls
    if (elements.lightboxClose) {
        elements.lightboxClose.addEventListener('click', closeLightbox);
    }
    if (elements.lightboxPrev) {
        elements.lightboxPrev.addEventListener('click', () => navigateLightbox(-1));
    }
    if (elements.lightboxNext) {
        elements.lightboxNext.addEventListener('click', () => navigateLightbox(1));
    }
    if (elements.lightboxDownload) {
        elements.lightboxDownload.addEventListener('click', downloadLightboxPhoto);
    }
    if (elements.lightbox) {
        // Close lightbox when clicking backdrop
        elements.lightbox.addEventListener('click', (e) => {
            if (e.target === elements.lightbox) {
                closeLightbox();
            }
        });
    }

    // Keyboard navigation for lightbox
    document.addEventListener('keydown', (e) => {
        if (currentLightboxIndex === -1) return;

        if (e.key === 'Escape') closeLightbox();
        if (e.key === 'ArrowLeft') navigateLightbox(-1);
        if (e.key === 'ArrowRight') navigateLightbox(1);
    });

    // Touch swipe navigation for lightbox (mobile)
    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', (e) => {
        if (currentLightboxIndex === -1) return;
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        if (currentLightboxIndex === -1) return;
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, { passive: true });

    function handleSwipe() {
        const swipeThreshold = 50; // Minimum swipe distance
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swiped left → next photo
                navigateLightbox(1);
            } else {
                // Swiped right → previous photo
                navigateLightbox(-1);
            }
        }
    }
});

// =============================================================================
// CAMERA FUNCTIONS
// =============================================================================

async function startCamera() {
    console.log('📷 Starting camera...');

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Camera is not supported on this device/browser. Please try uploading a photo instead.');
            console.error('getUserMedia not supported');
            return;
        }

        const constraints = {
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        };

        console.log('Requesting camera with constraints:', constraints);

        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('✓ Camera access granted');

        if (elements.cameraPreview) {
            elements.cameraPreview.srcObject = mediaStream;

            elements.cameraPreview.onloadedmetadata = () => {
                console.log('Video metadata loaded');
                elements.cameraPreview.play();
            };
        }

        if (elements.uploadSection) {
            elements.uploadSection.classList.add('hidden');
        }
        if (elements.cameraView) {
            elements.cameraView.classList.remove('hidden');
        }

        console.log('✓ Camera view shown');

    } catch (error) {
        console.error('Camera error:', error);

        if (error.name === 'NotAllowedError') {
            alert('Camera access denied. Please allow camera access in your browser settings and try again.');
        } else if (error.name === 'NotFoundError') {
            alert('No camera found. Please make sure your device has a camera or try uploading a photo instead.');
        } else if (error.name === 'NotReadableError') {
            alert('Camera is being used by another application. Please close other apps using the camera and try again.');
        } else if (error.name === 'OverconstrainedError') {
            console.log('Trying with simpler constraints...');
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
                elements.cameraPreview.srcObject = mediaStream;
                elements.uploadSection.classList.add('hidden');
                elements.cameraView.classList.remove('hidden');
                return;
            } catch (e) {
                alert('Could not access camera. Please try uploading a photo instead.');
            }
        } else {
            alert('Could not access camera: ' + error.message + '. Please try uploading a photo instead.');
        }
    }
}

function stopCamera() {
    console.log('Stopping camera...');

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            track.stop();
            console.log('Track stopped:', track.kind);
        });
        mediaStream = null;
    }

    if (elements.cameraPreview) {
        elements.cameraPreview.srcObject = null;
    }

    if (elements.cameraView) {
        elements.cameraView.classList.add('hidden');
    }
    if (elements.uploadSection) {
        elements.uploadSection.classList.remove('hidden');
    }

    console.log('✓ Camera stopped');
}

function capturePhoto() {
    console.log('📸 Capturing photo...');

    const video = elements.cameraPreview;
    if (!video || !video.videoWidth) {
        alert('Camera is not ready. Please wait a moment and try again.');
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        if (!blob) {
            alert('Failed to capture photo. Please try again.');
            return;
        }

        console.log('Photo captured, size:', blob.size);

        // Create a File object from the blob
        const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });

        // Add to selectedFiles array (supports multi-capture)
        selectedFiles.push(file);

        // Show preview grid with all captured/uploaded photos
        showMultiImagePreview(selectedFiles);

        // Stop camera but keep capture option available
        stopCamera();

        // Show upload section with preview
        if (elements.uploadSection) {
            elements.uploadSection.classList.remove('hidden');
        }
    }, 'image/jpeg', 0.9);
}

// =============================================================================
// FILE UPLOAD (MULTI-IMAGE SUPPORT)
// =============================================================================

// Store selected files for multi-image search
let selectedFiles = [];

async function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    if (!files.length) return;

    console.log('Files selected:', files.length);

    // Validate each file
    for (const file of files) {
        if (!file.type.startsWith('image/')) {
            alert('Please select only image files.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert(`File "${file.name}" is too large. Please select images smaller than 10MB.`);
            return;
        }
    }

    // Limit to 5 files
    const validFiles = files.slice(0, 5);
    if (files.length > 5) {
        alert('Maximum 5 images allowed. Only the first 5 will be used.');
    }

    // Store files
    selectedFiles = validFiles;

    // Show preview grid
    showMultiImagePreview(validFiles);

    event.target.value = ''; // Reset input
}

function showMultiImagePreview(files) {
    const previewSection = document.getElementById('multi-preview-section');
    const previewGrid = document.getElementById('multi-preview-grid');
    const previewCount = document.getElementById('preview-count');
    const searchBtn = document.getElementById('search-btn');
    const multiPhotoTip = document.getElementById('multi-photo-tip');

    if (!previewSection || !previewGrid) return;

    // Clear existing previews
    previewGrid.innerHTML = '';

    // Create preview for each file
    files.forEach((file, index) => {
        const url = URL.createObjectURL(file);
        const div = document.createElement('div');
        div.className = 'relative aspect-square rounded-lg overflow-hidden border-2 border-gold/30';
        div.innerHTML = `
            <img src="${url}" class="w-full h-full object-cover">
            <button onclick="removeSelectedPhoto(${index})" 
                    class="absolute top-1 right-1 w-5 h-5 bg-coral-pink text-white rounded-full text-xs flex items-center justify-center hover:bg-terracotta">
                ×
            </button>
        `;
        previewGrid.appendChild(div);
    });

    // Update count
    if (previewCount) {
        previewCount.textContent = files.length;
    }

    // Show tip for multi-photo
    if (multiPhotoTip) {
        multiPhotoTip.classList.toggle('hidden', files.length < 2);
    }

    // Show preview section and search button
    previewSection.classList.remove('hidden');
    if (searchBtn) {
        searchBtn.classList.remove('hidden');
        searchBtn.onclick = () => startSearchFromSelectedPhotos();
    }
}

function removeSelectedPhoto(index) {
    selectedFiles.splice(index, 1);
    if (selectedFiles.length === 0) {
        clearSelectedPhotos();
    } else {
        showMultiImagePreview(selectedFiles);
    }
}

function clearSelectedPhotos() {
    selectedFiles = [];
    const previewSection = document.getElementById('multi-preview-section');
    const searchBtn = document.getElementById('search-btn');

    if (previewSection) previewSection.classList.add('hidden');
    if (searchBtn) searchBtn.classList.add('hidden');
}

async function startSearchFromSelectedPhotos() {
    if (selectedFiles.length === 0) {
        alert('Please select at least one photo.');
        return;
    }

    console.log(`🔍 Starting search with ${selectedFiles.length} photo(s)...`);

    if (selectedFiles.length === 1) {
        // Single image: use original endpoint
        await submitSearch(selectedFiles[0]);
    } else {
        // Multiple images: use hybrid multi-image endpoint
        await submitMultiImageSearch(selectedFiles);
    }
}

async function submitMultiImageSearch(files) {
    console.log(`🔍 Submitting HYBRID multi-image search with ${files.length} photos...`);

    // Create preview URL from first image
    if (previewImageUrl) {
        URL.revokeObjectURL(previewImageUrl);
    }
    previewImageUrl = URL.createObjectURL(files[0]);

    showLoading(`Uploading ${files.length} photos...`, 1);

    const formData = new FormData();
    // Add all files with the same field name 'files'
    for (const file of files) {
        formData.append('files', file);
    }
    formData.append('threshold', elements.threshold ? elements.threshold.value : '0.55');

    try {
        const response = await fetch(`${API_BASE}/search-multi`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        currentTaskId = data.task_id;
        console.log('Multi-image search task created:', currentTaskId);

        showLoading(`Processing ${files.length} photos with hybrid matching...`, 2);
        pollForResults();

    } catch (error) {
        console.error('Multi-image search error:', error);
        showError(error.message);
    }
}

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function submitSearch(imageBlob) {
    console.log('🔍 Submitting search...');

    // Create preview URL for the uploaded/captured image
    if (previewImageUrl) {
        URL.revokeObjectURL(previewImageUrl);
    }
    previewImageUrl = URL.createObjectURL(imageBlob);

    showLoading('Uploading your photo...', 1);

    const formData = new FormData();
    formData.append('file', imageBlob, 'photo.jpg');
    formData.append('threshold', elements.threshold ? elements.threshold.value : '0.5');

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        currentTaskId = data.task_id;
        console.log('Search task created:', currentTaskId);

        showLoading('Detecting faces in your photo...', 2);
        pollForResults();

    } catch (error) {
        console.error('Search error:', error);
        showError(error.message);
    }
}

async function pollForResults() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`${API_BASE}/status/${currentTaskId}`);
        const data = await response.json();
        console.log('Poll result:', data.status);

        if (data.status === 'completed') {
            if (data.total_matches > 0) {
                showResults(data);
            } else {
                showNoResults();
            }
        } else if (data.status === 'failed') {
            showError(data.message || 'Search failed');
        } else {
            // Update to searching step after first poll
            showLoading('Searching through wedding photos...', 3);
            setTimeout(pollForResults, POLL_INTERVAL);
        }

    } catch (error) {
        console.error('Poll error:', error);
        showError('Failed to check search status');
    }
}

// =============================================================================
// UI STATE FUNCTIONS
// =============================================================================

function showLoading(message, step = 1) {
    hideAllSections();
    if (elements.loadingSection) {
        elements.loadingSection.classList.remove('hidden');
    }
    if (elements.loadingMessage) {
        elements.loadingMessage.textContent = message;
    }

    // Show preview photo
    if (elements.previewPhoto && previewImageUrl) {
        elements.previewPhoto.src = previewImageUrl;
    }

    // Update progress steps based on current step
    updateProgressSteps(step);

    // Update detail text based on step
    if (elements.loadingDetail) {
        const details = {
            1: 'Uploading your photo...',
            2: 'Looking for faces in your image',
            3: 'Comparing against wedding photos'
        };
        elements.loadingDetail.textContent = details[step] || '';
    }
}

function updateProgressSteps(currentStep) {
    // Step 2: Detecting faces
    if (elements.progressStep2) {
        const step2Circle = elements.progressStep2.querySelector('div');
        const step2Text = elements.progressStep2.querySelector('span:last-child');

        if (currentStep >= 2) {
            step2Circle.classList.remove('bg-gray-300', 'dark:bg-gray-600');
            step2Circle.classList.add('bg-gold', 'animate-pulse');
            step2Text.classList.remove('text-gray-400');
            step2Text.classList.add('text-gold');
        }
        if (currentStep > 2) {
            step2Circle.classList.remove('animate-pulse');
            step2Circle.classList.add('bg-emerald-green');
            step2Circle.innerHTML = '<span class="material-symbols-outlined text-white">check</span>';
            step2Text.classList.remove('text-gold');
            step2Text.classList.add('text-emerald-green');
        }
    }

    // Progress bar between step 2 and 3
    if (elements.progressBar2 && currentStep >= 3) {
        elements.progressBar2.classList.remove('bg-gray-300', 'dark:bg-gray-600');
        elements.progressBar2.classList.add('bg-emerald-green');
    }

    // Step 3: Searching
    if (elements.progressStep3) {
        const step3Circle = elements.progressStep3.querySelector('div');
        const step3Text = elements.progressStep3.querySelector('span:last-child');

        if (currentStep >= 3) {
            step3Circle.classList.remove('bg-gray-300', 'dark:bg-gray-600');
            step3Circle.classList.add('bg-gold', 'animate-pulse');
            step3Text.classList.remove('text-gray-400');
            step3Text.classList.add('text-gold');
        }
    }
}

function showResults(data) {
    hideAllSections();

    // Store results for filtering
    allPhotos = data.matches || [];
    unfilteredPhotos = [...allPhotos]; // Keep copy for filters
    selectedPhotos.clear();
    displayedCount = 0;

    // Clear any active filters for new search
    clearAllFilters();

    if (elements.resultCount) {
        elements.resultCount.textContent = data.total_matches;
    }
    if (elements.resultSubtitle) {
        elements.resultSubtitle.textContent = `Found ${data.face_count || 1} face(s) in your photo`;
    }

    // Clear grid and show initial batch
    if (elements.photosGrid) {
        elements.photosGrid.innerHTML = '';
    }
    showMorePhotos();
    updateDownloadButton();
    updateShowingCount();

    // Show query preview bar (persistent query images)
    showQueryPreview();

    if (elements.resultsSection) {
        elements.resultsSection.classList.remove('hidden');
    }
}

function showNoResults() {
    hideAllSections();
    if (elements.noResultsSection) {
        elements.noResultsSection.classList.remove('hidden');
    }
}

function showError(message) {
    hideAllSections();
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
    if (elements.errorSection) {
        elements.errorSection.classList.remove('hidden');
    }
}

function hideAllSections() {
    if (elements.loadingSection) elements.loadingSection.classList.add('hidden');
    if (elements.resultsSection) elements.resultsSection.classList.add('hidden');
    if (elements.noResultsSection) elements.noResultsSection.classList.add('hidden');
    if (elements.errorSection) elements.errorSection.classList.add('hidden');
}

function resetSearch() {
    hideAllSections();
    if (elements.uploadSection) {
        elements.uploadSection.classList.remove('hidden');
    }
    currentTaskId = null;
    selectedPhotos.clear();
    allPhotos = [];
    displayedCount = 0;
}

// =============================================================================
// PHOTO GRID WITH PAGINATION
// =============================================================================

function showMorePhotos() {
    if (!elements.photosGrid) return;

    const startIndex = displayedCount;
    const endIndex = Math.min(displayedCount + PHOTOS_PER_PAGE, allPhotos.length);

    for (let i = startIndex; i < endIndex; i++) {
        const card = createPhotoCard(allPhotos[i], i);
        elements.photosGrid.appendChild(card);
    }

    displayedCount = endIndex;
    updateShowingCount();
    updateSeeMoreButton();
}

function updateShowingCount() {
    if (elements.showingCount) {
        elements.showingCount.textContent = `Showing ${displayedCount} of ${allPhotos.length}`;
    }
}

function updateSeeMoreButton() {
    if (elements.seeMoreBtn) {
        if (displayedCount >= allPhotos.length) {
            elements.seeMoreBtn.classList.add('hidden');
        } else {
            elements.seeMoreBtn.classList.remove('hidden');
            const remaining = allPhotos.length - displayedCount;
            elements.seeMoreBtn.querySelector('span').textContent =
                `See More (${remaining} remaining)`;
        }
    }
}

function renderPhotos() {
    if (!elements.photosGrid) return;

    elements.photosGrid.innerHTML = '';
    displayedCount = 0;
    showMorePhotos();
}

function createPhotoCard(photo, index) {
    const card = document.createElement('div');
    card.className = 'photo-card relative rounded-xl overflow-hidden shadow-lg cursor-pointer border-2 border-gold/20';
    card.dataset.index = index;

    const isSelected = selectedPhotos.has(photo.image_id);

    card.innerHTML = `
        <div class="aspect-square">
            <img src="${photo.drive_thumbnail}" 
                 alt="Wedding photo" 
                 class="w-full h-full object-cover"
                 loading="lazy"
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%23eee%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2255%22 text-anchor=%22middle%22 fill=%22%23999%22>📷</text></svg>'">
        </div>
        
        <!-- Selection checkbox (top-right corner) -->
        <div class="absolute top-2 right-2 z-10" onclick="event.stopPropagation(); togglePhotoSelectionById('${photo.image_id}', this.parentElement);">
            <div class="w-7 h-7 rounded-full border-2 ${isSelected ? 'bg-gold border-gold' : 'bg-white/80 border-white'} 
                        flex items-center justify-center shadow-lg hover:scale-110 transition">
                ${isSelected ? '<span class="text-white text-sm">✓</span>' : ''}
            </div>
        </div>
        
        <!-- Expand button (center, appears on hover) -->
        <div class="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/20">
            <div class="w-12 h-12 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
                <span class="material-symbols-outlined text-peacock-blue">zoom_in</span>
            </div>
        </div>
        
        <!-- Info overlay (bottom) -->
        <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-3">
            <div class="flex items-center gap-2">
                <span class="text-white text-xs">${getEmotionEmoji(photo.emotions?.dominant)}</span>
                <span class="text-white text-xs opacity-75">${Math.round(photo.similarity * 100)}% match</span>
            </div>
        </div>
    `;

    // Click the main card to open lightbox
    card.addEventListener('click', (e) => {
        // Don't open lightbox if clicking checkbox
        if (e.target.closest('[onclick]')) return;
        openLightbox(index);
    });

    return card;
}

function getEmotionEmoji(emotion) {
    const emojis = {
        happy: '😊',
        sad: '😢',
        angry: '😠',
        surprise: '😮',
        fear: '😨',
        disgust: '🤢',
        neutral: '😐'
    };
    return emojis[emotion] || '📷';
}

// =============================================================================
// LIGHTBOX
// =============================================================================

function openLightbox(index) {
    if (!elements.lightbox || !allPhotos[index]) return;

    currentLightboxIndex = index;
    updateLightboxContent();
    elements.lightbox.classList.remove('hidden');
    document.body.style.overflow = 'hidden'; // Prevent scrolling
}

function closeLightbox() {
    if (!elements.lightbox) return;

    currentLightboxIndex = -1;
    elements.lightbox.classList.add('hidden');
    document.body.style.overflow = ''; // Restore scrolling
}

function navigateLightbox(direction) {
    const newIndex = currentLightboxIndex + direction;

    if (newIndex >= 0 && newIndex < allPhotos.length) {
        currentLightboxIndex = newIndex;
        updateLightboxContent();
    }
}

function updateLightboxContent() {
    const photo = allPhotos[currentLightboxIndex];
    if (!photo) return;

    // Use full-size view URL if available, otherwise use thumbnail
    const fullSizeUrl = photo.drive_view ||
        `https://drive.google.com/uc?export=view&id=${photo.image_id}`;

    if (elements.lightboxImage) {
        // Show loading state
        elements.lightboxImage.style.opacity = '0.5';

        const img = new Image();
        img.onload = () => {
            elements.lightboxImage.src = fullSizeUrl;
            elements.lightboxImage.style.opacity = '1';
        };
        img.onerror = () => {
            // Fallback to thumbnail if full size fails
            elements.lightboxImage.src = photo.drive_thumbnail;
            elements.lightboxImage.style.opacity = '1';
        };
        img.src = fullSizeUrl;
    }

    if (elements.lightboxCounter) {
        elements.lightboxCounter.textContent = `${currentLightboxIndex + 1} / ${allPhotos.length}`;
    }

    if (elements.lightboxInfo) {
        const emotion = photo.emotions?.dominant || 'neutral';
        const matchPercent = Math.round(photo.similarity * 100);
        elements.lightboxInfo.innerHTML = `
            <span class="mr-4">${getEmotionEmoji(emotion)} ${emotion}</span>
            <span>${matchPercent}% match</span>
        `;
    }

    // Update navigation button visibility
    if (elements.lightboxPrev) {
        elements.lightboxPrev.style.visibility = currentLightboxIndex > 0 ? 'visible' : 'hidden';
    }
    if (elements.lightboxNext) {
        elements.lightboxNext.style.visibility = currentLightboxIndex < allPhotos.length - 1 ? 'visible' : 'hidden';
    }
}

function downloadLightboxPhoto() {
    const photo = allPhotos[currentLightboxIndex];
    if (photo) {
        downloadPhotoFile(photo.drive_download, `wedding_photo_${photo.image_id}.jpg`);
    }
}

/**
 * Download a single photo file properly (works for local files)
 * Uses fetch + blob to ensure actual download instead of navigation
 */
async function downloadPhotoFile(url, filename) {
    try {
        // Fetch the file
        const response = await fetch(url);
        if (!response.ok) throw new Error('Download failed');

        const blob = await response.blob();

        // Create download link
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Clean up
        URL.revokeObjectURL(downloadUrl);
        console.log(`✓ Downloaded: ${filename}`);
    } catch (error) {
        console.error('Download failed:', error);
        // Fallback: open in new tab
        window.open(url, '_blank');
    }
}

// =============================================================================
// SELECTION
// =============================================================================

// Global function for onclick handler
window.togglePhotoSelectionById = function (imageId, cardElement) {
    if (selectedPhotos.has(imageId)) {
        selectedPhotos.delete(imageId);
    } else {
        selectedPhotos.add(imageId);
    }

    // Re-render the specific card
    const index = allPhotos.findIndex(p => p.image_id === imageId);
    if (index >= 0 && cardElement) {
        const newCard = createPhotoCard(allPhotos[index], index);
        cardElement.replaceWith(newCard);
    }

    updateDownloadButton();
};

function toggleSelectAll() {
    console.log('🔄 toggleSelectAll called, current selection count:', selectedPhotos.size);

    // If ANY photos are selected, deselect all. Otherwise select all.
    if (selectedPhotos.size > 0) {
        // Deselect all
        console.log('  → Clearing all selections');
        selectedPhotos.clear();
        updateSelectAllButtonText('Select All');
    } else {
        // Select all
        console.log('  → Selecting all', allPhotos.length, 'photos');
        allPhotos.forEach(p => selectedPhotos.add(p.image_id));
        updateSelectAllButtonText('Deselect All');
    }

    console.log('  → New selection count:', selectedPhotos.size);
    renderPhotos();
    updateDownloadButton();
}

function updateSelectAllButtonText(text) {
    if (elements.selectAllBtn) {
        // Get the second span (the text, not the icon)
        const spans = elements.selectAllBtn.querySelectorAll('span');
        if (spans.length >= 2) {
            spans[1].textContent = text;
        }
    }
}

function updateDownloadButton() {
    const count = selectedPhotos.size;
    if (elements.selectedCount) {
        elements.selectedCount.textContent = count;
    }
    // Also update the display count in the sticky toolbar
    const displayCount = document.getElementById('selected-count-display');
    if (displayCount) {
        displayCount.textContent = count;
    }
    if (elements.downloadSelectedBtn) {
        elements.downloadSelectedBtn.disabled = count === 0;
    }
    // Also update Share to Memories button
    if (elements.shareToMemoriesBtn) {
        elements.shareToMemoriesBtn.disabled = count === 0;
    }

    // Update button text to show current state
    updateSelectAllButtonText(count > 0 ? 'Deselect All' : 'Select All');
}

// =============================================================================
// DOWNLOAD
// =============================================================================

/**
 * Download selected photos.
 * 
 * NOTE: Google Drive has limitations:
 * - Direct download links open in new tabs
 * - Browsers block multiple window.open() calls (popup blocker)
 * - Rate limits may apply for bulk downloads
 * 
 * Solution: Download sequentially with user feedback, using hidden iframes
 * to avoid popup blockers.
 */
async function downloadSelected() {
    if (selectedPhotos.size === 0) {
        alert('Please select at least one photo to download.');
        return;
    }

    const selectedArray = Array.from(selectedPhotos);
    const total = selectedArray.length;
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;

    console.log(`📥 Starting download of ${total} photo(s)... (iOS: ${isIOS})`);

    // iOS: Try Web Share API for "Save to Photos" experience
    if (isIOS && total <= 5 && navigator.share && navigator.canShare) {
        try {
            await downloadForIOSShare(selectedArray);
            return;
        } catch (err) {
            console.log('Web Share failed, falling back to regular download', err);
        }
    }

    // For single photo, download immediately
    if (total === 1) {
        const photo = allPhotos.find(p => p.image_id === selectedArray[0]);
        if (photo) {
            await downloadPhotoFile(photo.drive_download, `wedding_photo_${photo.image_id}.jpg`);
        }
        return;
    }

    // ≤10 photos: Staggered individual downloads
    if (total <= 10) {
        await downloadStaggered(selectedArray);
        return;
    }

    // >10 photos: ZIP download
    await downloadAsZip(selectedArray);
}

/**
 * iOS Web Share API for native "Save to Photos" experience
 * Only works for small batches due to memory limits
 */
async function downloadForIOSShare(imageIds) {
    const files = [];

    for (const imageId of imageIds) {
        const photo = allPhotos.find(p => p.image_id === imageId);
        if (!photo) continue;

        try {
            const response = await fetch(photo.drive_download);
            const blob = await response.blob();
            const file = new File([blob], `wedding_photo_${imageId}.jpg`, { type: 'image/jpeg' });
            files.push(file);
        } catch (err) {
            console.error(`Failed to fetch ${imageId}:`, err);
        }
    }

    if (files.length > 0 && navigator.canShare({ files })) {
        await navigator.share({
            files: files,
            title: 'Wedding Photos',
            text: 'Save these photos to your gallery'
        });
        console.log('✓ Shared to iOS successfully');
    } else {
        throw new Error('Cannot share files');
    }
}

/**
 * Staggered individual downloads with delay (for ≤10 photos)
 * Browser-friendly approach that avoids download blocking
 */
async function downloadStaggered(imageIds) {
    const total = imageIds.length;
    const progressBar = showDownloadProgress(total);
    let completed = 0;
    let failed = 0;

    for (const imageId of imageIds) {
        const photo = allPhotos.find(p => p.image_id === imageId);
        if (!photo) {
            failed++;
            continue;
        }

        try {
            await downloadPhotoFile(photo.drive_download, `wedding_photo_${imageId}.jpg`);
            completed++;
            updateDownloadProgress(progressBar, completed, total);
        } catch (error) {
            failed++;
            console.error(`Failed to download ${imageId}:`, error);
        }

        // 300ms delay between downloads to prevent browser blocking
        if (completed < total) {
            await new Promise(resolve => setTimeout(resolve, 300));
        }
    }

    hideDownloadProgress(progressBar);
    console.log(`✓ Downloaded ${completed}/${total} photos${failed > 0 ? `, ${failed} failed` : ''}`);
}

/**
 * ZIP download for large batches (>10 photos)
 * Server creates ZIP file and sends as single download
 */
async function downloadAsZip(imageIds) {
    const progressBar = showDownloadProgress(imageIds.length);
    updateDownloadProgress(progressBar, 0, imageIds.length, 'Creating ZIP file...');

    try {
        // Build form data with image IDs
        const formData = new FormData();
        imageIds.forEach(id => formData.append('image_ids', id));

        // Request ZIP from server
        updateDownloadProgress(progressBar, 0, imageIds.length, 'Downloading from server...');
        const response = await fetch('/api/download-zip', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        updateDownloadProgress(progressBar, 0, imageIds.length, 'Preparing download...');

        // Get the blob
        const blob = await response.blob();

        // Hide progress before triggering download
        hideDownloadProgress(progressBar);

        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `wedding_photos_${imageIds.length}_images.zip`;
        document.body.appendChild(a);

        // Trigger download with small delay to allow UI to update
        await new Promise(resolve => setTimeout(resolve, 50));
        a.click();

        // Cleanup after a delay (don't revoke URL immediately)
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 1000);

        console.log(`✓ Downloaded ZIP with ${imageIds.length} photos`);

        // Show success message after download starts
        setTimeout(() => {
            alert(`Downloaded ${imageIds.length} photos as a ZIP file!\n\nCheck your Downloads folder.`);
        }, 500);

    } catch (error) {
        hideDownloadProgress(progressBar);
        console.error('ZIP download failed:', error);
        alert('Download failed. Please try selecting fewer photos or try again.');
    }
}

/**
 * Show download progress bar
 */
function showDownloadProgress(total) {
    const progressDiv = document.createElement('div');
    progressDiv.id = 'download-progress';
    progressDiv.className = 'fixed top-20 left-1/2 -translate-x-1/2 z-[200] bg-cream/95 dark:bg-peacock-blue/95 backdrop-blur-lg rounded-2xl p-4 shadow-2xl border-2 border-gold/30 min-w-[280px]';
    progressDiv.innerHTML = `
        <div class="flex items-center gap-3 mb-3">
            <span class="text-2xl animate-pulse">📥</span>
            <span class="font-heading text-lg text-peacock-blue dark:text-gold">Downloading...</span>
        </div>
        <div class="bg-champagne/50 dark:bg-peacock-blue/50 rounded-full h-3 overflow-hidden">
            <div id="download-progress-bar" class="h-full bg-gradient-to-r from-emerald-green to-teal-blue transition-all duration-300" style="width: 0%"></div>
        </div>
        <p id="download-progress-text" class="text-sm text-peacock-blue/70 dark:text-champagne/70 mt-2 text-center">
            0 / ${total} photos
        </p>
    `;
    document.body.appendChild(progressDiv);
    return progressDiv;
}

/**
 * Update download progress
 */
function updateDownloadProgress(progressDiv, completed, total, customMessage = null) {
    if (!progressDiv) return;
    const bar = progressDiv.querySelector('#download-progress-bar');
    const text = progressDiv.querySelector('#download-progress-text');
    const percent = Math.round((completed / total) * 100);
    if (bar) bar.style.width = `${percent}%`;
    if (text) text.textContent = customMessage || `${completed} / ${total} photos`;
}

/**
 * Hide download progress
 */
function hideDownloadProgress(progressDiv) {
    if (progressDiv) {
        progressDiv.remove();
    }
}

/**
 * Alternative: Download all selected as a list/info (for user to copy)
 * Useful if iframe method doesn't work on some browsers
 */
function getDownloadLinks() {
    const links = [];
    selectedPhotos.forEach(imageId => {
        const photo = allPhotos.find(p => p.image_id === imageId);
        if (photo) {
            links.push(photo.drive_download);
        }
    });
    return links;
}

// =============================================================================
// FILTER SYSTEM (Fun Features)
// =============================================================================

// Filter state
let activeFilters = {
    event: '',
    date: '',
    emotions: new Set(),
    withGroom: false,
    withBride: false
};

// Keep track of unfiltered results for re-filtering
let unfilteredPhotos = [];

/**
 * Toggle emotion filter button
 */
function toggleEmotionFilter(emotion) {
    const btn = document.querySelector(`[data-emotion="${emotion}"]`);

    if (activeFilters.emotions.has(emotion)) {
        activeFilters.emotions.delete(emotion);
        btn.classList.remove('bg-gold/40', 'ring-2', 'ring-gold');
    } else {
        activeFilters.emotions.add(emotion);
        btn.classList.add('bg-gold/40', 'ring-2', 'ring-gold');
    }

    applyFilters();
}

/**
 * Toggle bride/groom filter button
 * Uses has_bride/has_groom flags from API response
 */
function togglePersonFilter(person) {
    const btn = document.getElementById(`filter-${person}`);

    if (person === 'groom') {
        activeFilters.withGroom = !activeFilters.withGroom;
        btn.classList.toggle('bg-gold/40', activeFilters.withGroom);
        btn.classList.toggle('ring-2', activeFilters.withGroom);
        btn.classList.toggle('ring-gold', activeFilters.withGroom);
    } else {
        activeFilters.withBride = !activeFilters.withBride;
        btn.classList.toggle('bg-gold/40', activeFilters.withBride);
        btn.classList.toggle('ring-2', activeFilters.withBride);
        btn.classList.toggle('ring-gold', activeFilters.withBride);
    }

    applyFilters();
}

/**
 * Apply all active filters to the photo list
 */
function applyFilters() {
    // Get dropdown values
    const eventSelect = document.getElementById('filter-event');
    const dateSelect = document.getElementById('filter-date');

    activeFilters.event = eventSelect ? eventSelect.value : '';
    activeFilters.date = dateSelect ? dateSelect.value : '';

    // Filter the photos
    let filtered = unfilteredPhotos.filter(photo => {
        // Event filter
        if (activeFilters.event) {
            const eventValues = activeFilters.event.split(',');
            const photoEvent = photo.event || '';
            if (!eventValues.some(ev => photoEvent.toLowerCase().includes(ev.toLowerCase()))) {
                return false;
            }
        }

        // Date filter
        if (activeFilters.date) {
            const photoDate = photo.date || '';
            if (!photoDate.includes(activeFilters.date)) {
                return false;
            }
        }

        // Emotion filter (if any selected, photo must match one of them)
        if (activeFilters.emotions.size > 0) {
            // Emotions are stored as object with 'dominant' field: { happy: 0.9, ..., dominant: 'happy' }
            const photoEmotion = (photo.emotions?.dominant || photo.emotion || '').toLowerCase();
            if (!activeFilters.emotions.has(photoEmotion)) {
                return false;
            }
        }

        // Bride/Groom filter - uses has_bride/has_groom flags from API response
        if (activeFilters.withBride && !photo.has_bride) {
            return false;
        }
        if (activeFilters.withGroom && !photo.has_groom) {
            return false;
        }

        return true;
    });

    // Update display
    allPhotos = filtered;
    displayedCount = 0;

    // Clear and redisplay
    if (elements.photosGrid) {
        elements.photosGrid.innerHTML = '';
    }
    showMorePhotos();

    // Update counts
    const filteredCountEl = document.getElementById('filtered-count');
    const totalCountEl = document.getElementById('total-count');
    const activeFiltersEl = document.getElementById('active-filters');

    if (filteredCountEl) filteredCountEl.textContent = filtered.length;
    if (totalCountEl) totalCountEl.textContent = unfilteredPhotos.length;

    // Show active filters bar if any filter is active
    const hasFilters = activeFilters.event || activeFilters.date ||
        activeFilters.emotions.size > 0 ||
        activeFilters.withGroom || activeFilters.withBride;

    if (activeFiltersEl) {
        activeFiltersEl.classList.toggle('hidden', !hasFilters);
    }

    // Update result count in header
    if (elements.resultCount) {
        elements.resultCount.textContent = filtered.length;
    }
}

/**
 * Clear all filters
 */
function clearAllFilters() {
    // Reset state
    activeFilters = {
        event: '',
        date: '',
        emotions: new Set(),
        withGroom: false,
        withBride: false
    };

    // Reset dropdowns
    const eventSelect = document.getElementById('filter-event');
    const dateSelect = document.getElementById('filter-date');
    if (eventSelect) eventSelect.value = '';
    if (dateSelect) dateSelect.value = '';

    // Reset emotion buttons
    document.querySelectorAll('.emotion-btn').forEach(btn => {
        btn.classList.remove('bg-gold/40', 'ring-2', 'ring-gold');
    });

    // Reset person buttons
    document.querySelectorAll('.person-btn').forEach(btn => {
        btn.classList.remove('bg-gold/40', 'ring-2', 'ring-gold');
    });

    // Restore all photos
    allPhotos = [...unfilteredPhotos];
    displayedCount = 0;

    // Redisplay
    if (elements.photosGrid) {
        elements.photosGrid.innerHTML = '';
    }
    showMorePhotos();

    // Hide active filters bar
    const activeFiltersEl = document.getElementById('active-filters');
    if (activeFiltersEl) activeFiltersEl.classList.add('hidden');

    // Update header count
    if (elements.resultCount) {
        elements.resultCount.textContent = unfilteredPhotos.length;
    }
}

// =============================================================================
// QUERY PREVIEW (Persistent query images)
// =============================================================================

/**
 * Show query images in the results section so they stay visible
 */
function showQueryPreview() {
    const previewBar = document.getElementById('query-preview-bar');
    const previewImages = document.getElementById('query-preview-images');

    if (!previewBar || !previewImages || selectedFiles.length === 0) return;

    // Clear existing previews
    previewImages.innerHTML = '';

    // Add each query image
    selectedFiles.forEach((file, index) => {
        const url = URL.createObjectURL(file);
        const img = document.createElement('img');
        img.src = url;
        img.className = 'w-12 h-12 object-cover rounded-lg border-2 border-gold/30';
        img.alt = `Query image ${index + 1}`;
        previewImages.appendChild(img);
    });

    // Show the bar
    previewBar.classList.remove('hidden');
}

// =============================================================================
// SHARE TO MEMORIES
// =============================================================================

/**
 * Open share modal to collect optional name/message before sharing
 */
function openShareModal() {
    if (selectedPhotos.size === 0) {
        alert('Please select at least one photo to share.');
        return;
    }

    // Create modal HTML
    const modalHtml = `
        <div id="share-modal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-[110] flex items-center justify-center p-4">
            <div class="bg-cream dark:bg-peacock-blue rounded-2xl p-6 max-w-md w-full shadow-2xl border-2 border-gold/30">
                <div class="flex items-center gap-3 mb-6">
                    <div class="text-4xl">💕</div>
                    <h3 class="font-heading text-2xl font-bold text-peacock-blue dark:text-gold">
                        Share ${selectedPhotos.size} Photo${selectedPhotos.size > 1 ? 's' : ''}
                    </h3>
                </div>

                <div class="space-y-4 mb-6">
                    <div>
                        <label class="block font-elegant text-sm text-peacock-blue dark:text-champagne mb-2">
                            Your Name (Optional)
                        </label>
                        <input type="text" id="share-author-name" placeholder="Anonymous" 
                            class="w-full border-2 border-champagne dark:border-peacock-blue/50 rounded-xl px-4 py-3 font-body text-sm bg-white/80 dark:bg-peacock-blue/30 text-peacock-blue dark:text-champagne focus:border-emerald-green focus:outline-none">
                    </div>
                    <div>
                        <label class="block font-elegant text-sm text-peacock-blue dark:text-champagne mb-2">
                            Message (Optional)
                        </label>
                        <textarea id="share-message" rows="3" placeholder="Add a memory or message..."
                            class="w-full border-2 border-champagne dark:border-peacock-blue/50 rounded-xl px-4 py-3 font-body text-sm bg-white/80 dark:bg-peacock-blue/30 text-peacock-blue dark:text-champagne focus:border-emerald-green focus:outline-none resize-none"></textarea>
                    </div>
                </div>

                <div class="flex gap-3">
                    <button onclick="closeShareModal()" 
                        class="flex-1 bg-champagne dark:bg-peacock-blue/50 text-peacock-blue dark:text-champagne font-body font-semibold py-3 rounded-full transition-all hover:bg-champagne/80">
                        Cancel
                    </button>
                    <button onclick="shareToMemories()" 
                        class="flex-1 bg-gradient-to-r from-gold to-warm-brass text-white font-body font-semibold py-3 rounded-full shadow-lg hover:shadow-xl transition-all transform hover:scale-105">
                        Share to Memories
                    </button>
                </div>
            </div>
        </div>
    `;

    // Add modal to DOM
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

/**
 * Close the share modal
 */
window.closeShareModal = function () {
    const modal = document.getElementById('share-modal');
    if (modal) {
        modal.remove();
    }
};

/**
 * Share selected photos to Memories page
 */
window.shareToMemories = async function () {
    const authorName = document.getElementById('share-author-name')?.value.trim() || 'Anonymous';
    const message = document.getElementById('share-message')?.value.trim() || '';

    const imageIds = Array.from(selectedPhotos);

    try {
        const formData = new FormData();
        // Add each image ID separately (FastAPI expects multiple values for list)
        imageIds.forEach(id => formData.append('image_ids', id));
        if (message) formData.append('message', message);
        formData.append('author_name', authorName);

        const response = await fetch('/api/gallery/share-wedding-photos', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            closeShareModal();
            alert(`🎉 ${result.message}\n\nYour photos are now on the Memories page!`);

            // Clear selection after sharing
            selectedPhotos.clear();
            updateDownloadButton();
            renderPhotos();
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Share failed');
        }
    } catch (error) {
        console.error('Share error:', error);
        alert('Failed to share photos. Please try again.');
    }
};

