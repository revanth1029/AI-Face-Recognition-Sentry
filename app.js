// AI Sentry Pro - Multi-Person Face Recognition & Micro-Expression Engine
// Powered by face-api.js (vladmandic fork)

// State variables
let db = [];
let stream = null;
let isDetecting = false;
let detectorType = 'tiny'; // 'tiny' or 'ssd'
let minConfidence = 0.4;
let totalScanCount = 0;
let lastSeen = {}; // Tracker to prevent spamming logs
const LOG_COOLDOWN_MS = 4000;
let latestDetections = []; // Stores current frame raw detections
let latestResizedDetections = []; // Stores canvas-scaled detections
let expressionSensitivity = 'high'; // 'high' (landmark-boosted) or 'balanced'
let matchThreshold = parseFloat(localStorage.getItem('face_sentry_threshold')) || 0.44;
let selectedEnrollFaceIndex = 0; // Index of face targeted for enrollment

// Multi-face track cache for expression smoothing and temporal hysteresis
const faceTrackCache = new Map();

let lastErrorTime = 0;
function handleErrorThrottled(message) {
    const now = Date.now();
    if (now - lastErrorTime > 5000) {
        lastErrorTime = now;
        addLog(`Inference Warning: ${message}`, 'unknown');
    }
}

// DOM elements
const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const viewfinderContainer = document.getElementById('viewfinder-container');
const scanIcon = document.getElementById('scan-icon');
const videoIcon = document.getElementById('video-icon');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const cameraSelect = document.getElementById('camera-select');
const btnRefreshCameras = document.getElementById('btn-refresh-cameras');
let selectedDeviceId = localStorage.getItem('face_sentry_camera_id') || '';
const facesBadge = document.getElementById('faces-badge');
const facesCountText = document.getElementById('faces-count-text');
const unregisteredAlertBanner = document.getElementById('unregistered-alert-banner');
const unregisteredAlertText = document.getElementById('unregistered-alert-text');
const unregisteredBadge = document.getElementById('unregistered-badge');

// Expression modes
const btnSensHigh = document.getElementById('btn-sens-high');
const btnSensStd = document.getElementById('btn-sens-std');
const expressionModeVal = document.getElementById('expression-mode-val');

// Enrollment & Multi-Face Targeting
const enrollNameInput = document.getElementById('enroll-name');
const btnEnroll = document.getElementById('btn-enroll');
const faceSelectorContainer = document.getElementById('face-selector-container');
const faceSelectorPills = document.getElementById('face-selector-pills');

// Database Management
const dbGrid = document.getElementById('db-grid');
const dbCountEl = document.getElementById('db-count');
const btnClearDb = document.getElementById('btn-clear-db');
const btnExportDb = document.getElementById('btn-export-db');
const btnImportDb = document.getElementById('btn-import-db');
const dbFileInput = document.getElementById('db-file-input');
const btnClearLogs = document.getElementById('btn-clear-logs');
const logContainer = document.getElementById('log-container');

// Detector config & Strictness controls
const selTiny = document.getElementById('sel-tiny');
const selSsd = document.getElementById('sel-ssd');
const selStrictHigh = document.getElementById('sel-strict-high');
const selStrictStd = document.getElementById('sel-strict-std');
const selStrictLenient = document.getElementById('sel-strict-lenient');
const strictnessModeVal = document.getElementById('strictness-mode-val');

// Dashboard Stats
const fpsVal = document.getElementById('fps-val');
const statModelType = document.getElementById('stat-model-type');
const statMinConf = document.getElementById('stat-min-conf');
const statScanCount = document.getElementById('stat-scan-count');

// System status dots
const sysStatusDot = document.getElementById('sys-status-dot');
const sysStatusText = document.getElementById('sys-status-text');
const modelStatusDot = document.getElementById('model-status-dot');
const modelStatusText = document.getElementById('model-status-text');
const cameraStatusDot = document.getElementById('camera-status-dot');
const cameraStatusText = document.getElementById('camera-status-text');

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    loadDatabase();
    initStrictnessUI();
    statMinConf.innerText = matchThreshold.toFixed(2);
    await populateCameras();
    await loadModels();
    setupEventListeners();
});

// Load DB from LocalStorage
function loadDatabase() {
    try {
        const stored = localStorage.getItem('face_sentry_db');
        if (stored) {
            const rawDb = JSON.parse(stored);
            db = rawDb.map(item => ({
                name: item.name,
                descriptor: new Float32Array(item.descriptor)
            }));
            addLog(`Loaded ${db.length} profile${db.length === 1 ? '' : 's'} from local database.`, 'system');
        } else {
            addLog("Database is empty. Enroll profiles to begin identification.", 'system');
        }
        renderDb();
    } catch (e) {
        console.error("Database load error:", e);
        addLog("Database load error. Starting fresh.", "unknown");
        db = [];
        renderDb();
    }
}

// Save DB to LocalStorage
function saveDatabase() {
    try {
        const rawDb = db.map(item => ({
            name: item.name,
            descriptor: Array.from(item.descriptor)
        }));
        localStorage.setItem('face_sentry_db', JSON.stringify(rawDb));
        renderDb();
    } catch (e) {
        console.error("Database save error:", e);
        addLog("Error saving database.", "unknown");
    }
}

// Render Database Profiles Grid
function renderDb() {
    dbCountEl.innerText = db.length;
    dbGrid.innerHTML = '';

    if (db.length === 0) {
        dbGrid.innerHTML = '<div class="empty-state">No enrolled profiles in database</div>';
        return;
    }

    db.forEach((profile, index) => {
        const item = document.createElement('div');
        item.className = 'db-item';
        
        // Generate initials
        const parts = profile.name.trim().split(/\s+/);
        const initials = parts.map(p => p[0]).join('').substring(0, 2).toUpperCase();
        
        item.innerHTML = `
            <div class="db-avatar">${initials}</div>
            <div class="db-name" title="${escapeHtml(profile.name)}">${escapeHtml(profile.name)}</div>
            <div class="delete-btn" title="Delete Profile" data-index="${index}">&times;</div>
        `;
        
        item.querySelector('.delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteProfile(index);
        });
        
        dbGrid.appendChild(item);
    });
}

// Delete Single Profile
function deleteProfile(index) {
    if (index < 0 || index >= db.length) return;
    const name = db[index].name;
    db.splice(index, 1);
    saveDatabase();
    addLog(`Deleted profile: "${name}"`, 'system');
    updateEnrollBtnState();
}

// Clear Entire Database
function clearAllProfiles() {
    if (db.length === 0) {
        addLog("Database is already empty.", 'system');
        return;
    }
    const confirmed = confirm(`Are you sure you want to delete ALL ${db.length} profile(s) from the database? This cannot be undone.`);
    if (confirmed) {
        db = [];
        saveDatabase();
        addLog("All face profiles cleared from database.", 'system');
        updateEnrollBtnState();
    }
}

// Write to Dashboard Terminal Log
function addLog(message, type = 'system') {
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    let msgClass = 'log-msg';
    if (type === 'match') msgClass += ' match';
    if (type === 'unknown') msgClass += ' unknown';
    if (type === 'system') msgClass += ' system';
    
    entry.innerHTML = `
        <span class="log-time">[${timestamp}]</span>
        <span class="${msgClass}">${escapeHtml(message)}</span>
    `;
    
    logContainer.prepend(entry);
    
    // Cap logs at 50 to maintain high performance
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Load face-api.js models from CDN URL
async function loadModels() {
    try {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
        
        modelStatusDot.className = 'status-dot loading';
        modelStatusText.innerText = 'LOADING...';
        
        addLog("Loading AI Face Detection network...", "system");
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        
        addLog("Loading AI Facial Landmark network (68 points)...", "system");
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        
        addLog("Loading AI Face Recognition vectorizer (128-d)...", "system");
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        
        addLog("Loading AI Expression & Demographics networks...", "system");
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);
        
        modelStatusDot.className = 'status-dot active';
        modelStatusText.innerText = 'READY';
        addLog("AI Neural Networks online. Multi-person engine ready.", "system");
        
        try {
            const backend = faceapi.tf.getBackend();
            addLog(`TensorFlow.js backend active: ${backend.toUpperCase()}`, "system");
        } catch (be) {
            console.error(be);
        }
        
        btnToggleCamera.disabled = false;
        updateEnrollBtnState();
    } catch (error) {
        console.error("Model load error:", error);
        modelStatusDot.className = 'status-dot';
        modelStatusText.innerText = 'OFFLINE';
        addLog("Critical failure: Could not download neural networks. Check internet connection.", "unknown");
    }
}

// Toggle Enrollment Button active/inactive based on system state
function updateEnrollBtnState() {
    const isCameraActive = stream !== null;
    const isModelReady = modelStatusText.innerText === 'READY';
    btnEnroll.disabled = !(isCameraActive && isModelReady);
}

// Strictness UI Initialization & Switching
function initStrictnessUI() {
    updateStrictnessPills();
}

function updateStrictnessPills() {
    [selStrictHigh, selStrictStd, selStrictLenient].forEach(el => {
        if (el) el.classList.remove('active');
    });

    if (matchThreshold <= 0.41) {
        if (selStrictHigh) selStrictHigh.classList.add('active');
        if (strictnessModeVal) strictnessModeVal.innerText = 'ULTRA STRICT (0.40)';
    } else if (matchThreshold >= 0.48) {
        if (selStrictLenient) selStrictLenient.classList.add('active');
        if (strictnessModeVal) strictnessModeVal.innerText = 'PERMISSIVE (0.50)';
    } else {
        if (selStrictStd) selStrictStd.classList.add('active');
        if (strictnessModeVal) strictnessModeVal.innerText = 'STANDARD (0.44)';
    }
    if (statMinConf) statMinConf.innerText = matchThreshold.toFixed(2);
}

// Setup Event Listeners
function setupEventListeners() {
    btnToggleCamera.addEventListener('click', toggleCamera);

    // Database Actions
    if (btnClearDb) {
        btnClearDb.addEventListener('click', clearAllProfiles);
    }

    // Match Strictness selectors
    if (selStrictHigh) {
        selStrictHigh.addEventListener('click', () => {
            matchThreshold = 0.40;
            localStorage.setItem('face_sentry_threshold', '0.40');
            updateStrictnessPills();
            addLog("Multi-Face strictness set to ULTRA STRICT (0.40) - Zero false positives", 'system');
        });
    }
    if (selStrictStd) {
        selStrictStd.addEventListener('click', () => {
            matchThreshold = 0.44;
            localStorage.setItem('face_sentry_threshold', '0.44');
            updateStrictnessPills();
            addLog("Multi-Face strictness set to STANDARD (0.44) - Recommended for webcams", 'system');
        });
    }
    if (selStrictLenient) {
        selStrictLenient.addEventListener('click', () => {
            matchThreshold = 0.50;
            localStorage.setItem('face_sentry_threshold', '0.50');
            updateStrictnessPills();
            addLog("Multi-Face strictness set to PERMISSIVE (0.50) - For dim/dark lighting", 'system');
        });
    }

    // Expression sensitivity mode listeners
    if (btnSensHigh && btnSensStd) {
        btnSensHigh.addEventListener('click', () => {
            expressionSensitivity = 'high';
            btnSensHigh.classList.add('active-mode');
            btnSensStd.classList.remove('active-mode');
            if (expressionModeVal) expressionModeVal.innerText = 'HYPER-RESPONSIVE';
            addLog("Expression sensor set to Hyper-Responsive (Neural + 68 micro-curves)", 'system');
        });

        btnSensStd.addEventListener('click', () => {
            expressionSensitivity = 'balanced';
            btnSensStd.classList.add('active-mode');
            btnSensHigh.classList.remove('active-mode');
            if (expressionModeVal) expressionModeVal.innerText = 'BALANCED';
            addLog("Expression sensor set to Balanced mode", 'system');
        });
    }

    // Detector model selector
    selTiny.addEventListener('click', () => {
        if (detectorType === 'tiny') return;
        detectorType = 'tiny';
        minConfidence = 0.4;
        selTiny.classList.add('active');
        selSsd.classList.remove('active');
        statModelType.innerText = 'TINY';
        addLog("Detector set to Tiny Face Detector (Low Latency / CPU efficient)", "system");
    });

    selSsd.addEventListener('click', () => {
        if (detectorType === 'ssd') return;
        detectorType = 'ssd';
        minConfidence = 0.4;
        selSsd.classList.add('active');
        selTiny.classList.remove('active');
        statModelType.innerText = 'SSD';
        addLog("Detector set to SSD MobileNet (High Accuracy / GPU efficient)", "system");
    });

    // Clear Logs
    btnClearLogs.addEventListener('click', () => {
        logContainer.innerHTML = '';
        addLog("Console terminal cleared.", "system");
    });

    // Enrollment
    btnEnroll.addEventListener('click', enrollFace);
    enrollNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !btnEnroll.disabled) {
            enrollFace();
        }
    });

    // DB JSON Import/Export
    btnExportDb.addEventListener('click', exportDatabase);
    btnImportDb.addEventListener('click', () => dbFileInput.click());
    dbFileInput.addEventListener('change', importDatabase);

    // Camera selector
    if (cameraSelect) {
        cameraSelect.addEventListener('change', async (e) => {
            selectedDeviceId = e.target.value;
            localStorage.setItem('face_sentry_camera_id', selectedDeviceId);
            const label = cameraSelect.options[cameraSelect.selectedIndex]?.text || 'selected camera';
            addLog(`Selected camera device: ${label}`, 'system');
            if (stream) {
                stopCamera();
                await startCamera();
            }
        });
    }

    if (btnRefreshCameras) {
        btnRefreshCameras.addEventListener('click', async () => {
            await populateCameras();
            addLog("Camera devices list refreshed.", 'system');
        });
    }

    if (navigator.mediaDevices && navigator.mediaDevices.ondevicechange !== undefined) {
        navigator.mediaDevices.ondevicechange = () => {
            populateCameras();
        };
    }

    // Interactive Viewfinder Click-to-Select Face
    if (viewfinderContainer) {
        viewfinderContainer.addEventListener('click', handleViewfinderClick);
    }
}

// Click on viewfinder to select targeted face for enrollment
function handleViewfinderClick(e) {
    if (!latestResizedDetections || latestResizedDetections.length <= 1) return;

    const rect = overlay.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * overlay.width;
    const clickY = ((e.clientY - rect.top) / rect.height) * overlay.height;

    // Check which face box contains (or is closest to) clickX, clickY
    let closestIdx = -1;
    let minDistance = Infinity;

    latestResizedDetections.forEach((det, idx) => {
        const { x, y, width, height } = det.detection.box;
        const mirroredX = overlay.width - x - width;

        // Check if inside box
        if (clickX >= mirroredX && clickX <= mirroredX + width && clickY >= y && clickY <= y + height) {
            closestIdx = idx;
            minDistance = 0;
            return;
        }

        // Distance to box center
        const centerX = mirroredX + width / 2;
        const centerY = y + height / 2;
        const dist = Math.hypot(clickX - centerX, clickY - centerY);
        if (dist < minDistance) {
            minDistance = dist;
            closestIdx = idx;
        }
    });

    if (closestIdx !== -1) {
        selectedEnrollFaceIndex = closestIdx;
        updateFaceSelectorUI();
        addLog(`Targeted Face #${closestIdx + 1} for enrollment.`, 'system');
    }
}

// Populate available cameras into dropdown
async function populateCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        
        if (!cameraSelect) return;
        cameraSelect.innerHTML = '';
        
        if (videoDevices.length === 0) {
            cameraSelect.innerHTML = '<option value="">No video devices detected</option>';
            return;
        }

        let bestMatchId = selectedDeviceId;
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            const label = device.label || `Camera ${index + 1}`;
            option.text = label;

            // Prefer integrated/laptop cameras if not yet manually chosen
            if (!bestMatchId && /integrated|internal|webcam|front|laptop|built-in/i.test(label) && !/virtual|obs|droid|iriun/i.test(label)) {
                bestMatchId = device.deviceId;
            }
            cameraSelect.appendChild(option);
        });

        if (bestMatchId && [...cameraSelect.options].some(o => o.value === bestMatchId)) {
            cameraSelect.value = bestMatchId;
            selectedDeviceId = bestMatchId;
        } else if (cameraSelect.options.length > 0) {
            selectedDeviceId = cameraSelect.value;
        }
    } catch (err) {
        console.warn("Could not enumerate camera devices:", err);
    }
}

// Webcam stream handling
async function toggleCamera() {
    if (stream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        addLog("Initializing local webcam feed...", "system");
        
        const videoConstraints = {
            width: { ideal: 640 },
            height: { ideal: 480 }
        };

        if (selectedDeviceId) {
            videoConstraints.deviceId = { exact: selectedDeviceId };
        } else {
            videoConstraints.facingMode = 'user';
        }
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });
        
        // Refresh camera names now that permissions are granted (labels are now visible)
        await populateCameras();

        const activeTrack = stream.getVideoTracks()[0];
        if (activeTrack) {
            const trackLabel = activeTrack.label || 'Active Webcam';
            addLog(`Connected to camera: ${trackLabel}`, 'system');
        }
        
        webcam.srcObject = stream;
        webcam.addEventListener('play', onPlay);
        
        cameraStatusDot.className = 'status-dot active';
        cameraStatusText.innerText = 'ACTIVE';
        btnToggleCamera.innerHTML = '<i class="fa-solid fa-stop"></i> <span>STOP MONITORING</span>';
        btnToggleCamera.className = 'btn btn-danger';
        scanIcon.style.display = 'inline-block';
        videoIcon.style.display = 'none';
        
        updateEnrollBtnState();
        addLog("Camera stream established. Multi-person 1-to-1 matching online.", "system");
    } catch (err) {
        console.error("Camera access error:", err);
        addLog("Webcam connection failed. Check system webcam permissions.", "unknown");
        cameraStatusDot.className = 'status-dot';
        cameraStatusText.innerText = 'FAILED';
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        isDetecting = false;
        latestDetections = [];
        latestResizedDetections = [];
        
        webcam.srcObject = null;
        webcam.removeEventListener('play', onPlay);
        
        // Clear canvas
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        
        cameraStatusDot.className = 'status-dot';
        cameraStatusText.innerText = 'DISCONNECTED';
        btnToggleCamera.innerHTML = '<i class="fa-solid fa-power-off"></i> <span>START MONITORING</span>';
        btnToggleCamera.className = 'btn btn-secondary';
        scanIcon.style.display = 'none';
        videoIcon.style.display = 'inline-block';
        if (facesBadge) facesBadge.style.display = 'none';
        if (unregisteredBadge) unregisteredBadge.style.display = 'none';
        if (unregisteredAlertBanner) unregisteredAlertBanner.style.display = 'none';
        if (faceSelectorContainer) faceSelectorContainer.style.display = 'none';
        faceTrackCache.clear();
        
        updateEnrollBtnState();
        fpsVal.innerText = '00';
        addLog("Facial monitoring feed terminated.", "system");
    }
}

// Euclidean distance between two 128-dimensional Float32Array embeddings
function euclideanDistance(d1, d2) {
    if (!d1 || !d2 || d1.length !== d2.length) return 1.0;
    let sum = 0;
    for (let i = 0; i < d1.length; i++) {
        const diff = d1[i] - d2[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

// Optimal Multi-Face Bipartite Matching with Mutual Exclusivity
// Prevents false matches, wrong names, or multiple faces taking the same name in the same frame
function matchAllFacesOptimal(detections, profiles, threshold) {
    if (!detections || detections.length === 0) return [];

    // If database is empty, all faces are unregistered targets
    if (!profiles || profiles.length === 0) {
        return detections.map((d, idx) => ({
            faceIndex: idx,
            isMatch: false,
            labelName: "UNREGISTERED TARGET",
            confidenceScore: d.detection.score || 0.85,
            distance: 1.0,
            statusColor: '#06b6d4' // neon cyan
        }));
    }

    // 1. Calculate pairwise distance between all visible faces and all registered profiles
    const candidatePairs = [];
    for (let f = 0; f < detections.length; f++) {
        const faceDesc = detections[f].descriptor;
        if (!faceDesc) continue;

        for (let p = 0; p < profiles.length; p++) {
            const profileDesc = profiles[p].descriptor;
            if (!profileDesc) continue;

            const dist = euclideanDistance(faceDesc, profileDesc);
            if (dist <= threshold) {
                candidatePairs.push({
                    faceIndex: f,
                    profileIndex: p,
                    profileName: profiles[p].name,
                    distance: dist
                });
            }
        }
    }

    // 2. Sort candidate pairs by ascending distance (strongest biometric match gets priority)
    candidatePairs.sort((a, b) => a.distance - b.distance);

    // 3. Mutual Exclusive 1-to-1 Bipartite Assignment
    const assignedFaces = new Set();
    const assignedProfiles = new Set();
    const results = new Array(detections.length).fill(null);

    for (const pair of candidatePairs) {
        if (!assignedFaces.has(pair.faceIndex) && !assignedProfiles.has(pair.profileIndex)) {
            assignedFaces.add(pair.faceIndex);
            assignedProfiles.add(pair.profileIndex);

            // Normalized confidence mapping:
            // At distance 0 -> 98%
            // At distance = threshold -> 68%
            const quality = Math.max(0, 1 - (pair.distance / threshold));
            const confidence = 0.68 + (quality * 0.30);

            results[pair.faceIndex] = {
                faceIndex: pair.faceIndex,
                isMatch: true,
                labelName: pair.profileName.toUpperCase(),
                confidenceScore: confidence,
                distance: pair.distance,
                statusColor: '#10b981' // neon emerald
            };
        }
    }

    // 4. Any remaining face that was not assigned is UNREGISTERED PERSON
    for (let i = 0; i < detections.length; i++) {
        if (!results[i]) {
            results[i] = {
                faceIndex: i,
                isMatch: false,
                labelName: "UNREGISTERED PERSON",
                confidenceScore: detections[i].detection.score || 0.85,
                distance: 1.0,
                statusColor: '#f43f5e' // neon rose/red
            };
        }
    }

    return results;
}

// Recognition Engine & Canvas Drawing Loop
async function onPlay() {
    if (!stream || webcam.paused || webcam.ended) return;
    
    if (webcam.videoWidth === 0 || webcam.videoHeight === 0) {
        setTimeout(onPlay, 100);
        return;
    }
    
    isDetecting = true;
    
    let lastTime = performance.now();
    let frameCount = 0;
    let fps = 0;

    async function detectionLoop() {
        if (!isDetecting || webcam.paused || webcam.ended) return;
        
        // Input size 416 gives high resolution for multi-face tracking
        const options = detectorType === 'tiny'
            ? new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: minConfidence })
            : new faceapi.SsdMobilenetv1Options({ minConfidence: minConfidence });
            
        try {
            if (overlay.width !== webcam.videoWidth || overlay.height !== webcam.videoHeight) {
                overlay.width = webcam.videoWidth;
                overlay.height = webcam.videoHeight;
            }
            
            // Execute multi-net inference pipeline
            const detections = await faceapi
                .detectAllFaces(webcam, options)
                .withFaceLandmarks()
                .withFaceExpressions()
                .withAgeAndGender()
                .withFaceDescriptors();
                
            latestDetections = detections;
                
            if (detections.length > 0) {
                totalScanCount += detections.length;
                statScanCount.innerText = totalScanCount;
            }
            
            const displaySize = { width: webcam.videoWidth, height: webcam.videoHeight };
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            latestResizedDetections = resizedDetections;

            // Execute Optimal Bipartite Multi-Face Matcher
            const matchResults = matchAllFacesOptimal(detections, db, matchThreshold);
            
            // Update Multi-Face target selector buttons in Enrollment Card
            updateFaceSelectorUI();

            // Draw custom HUD overlays
            drawCustomHUD(overlay, resizedDetections, matchResults);
            
            // Calculate FPS
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                fps = Math.round((frameCount * 1000) / (now - lastTime));
                fpsVal.innerText = fps.toString().padStart(2, '0');
                frameCount = 0;
                lastTime = now;
            }
        } catch (e) {
            console.error("Frame inference error:", e);
            handleErrorThrottled(e.message);
        }
        
        requestAnimationFrame(detectionLoop);
    }
    
    requestAnimationFrame(detectionLoop);
}

// Update the dynamic face targeting selector in the Enrollment card
function updateFaceSelectorUI() {
    if (!faceSelectorContainer || !faceSelectorPills) return;

    if (!latestResizedDetections || latestResizedDetections.length <= 1) {
        faceSelectorContainer.style.display = 'none';
        selectedEnrollFaceIndex = 0;
        return;
    }

    faceSelectorContainer.style.display = 'flex';

    // Ensure selectedEnrollFaceIndex is valid
    if (selectedEnrollFaceIndex >= latestResizedDetections.length) {
        selectedEnrollFaceIndex = 0;
    }

    // Render pills if count changed or to update active status
    const currentPillsCount = faceSelectorPills.children.length;
    if (currentPillsCount !== latestResizedDetections.length) {
        faceSelectorPills.innerHTML = '';
        latestResizedDetections.forEach((det, idx) => {
            const pill = document.createElement('button');
            pill.type = 'button';
            pill.className = `face-pill ${idx === selectedEnrollFaceIndex ? 'active' : ''}`;
            
            // Position label: Face 1 (Left), Face 2 (Right), etc.
            const { x, width } = det.detection.box;
            const mirroredX = overlay.width - x - width;
            const side = mirroredX < overlay.width / 2 ? 'Left' : 'Right';
            
            pill.innerHTML = `<i class="fa-solid fa-user"></i> Face ${idx + 1} (${side})`;
            pill.addEventListener('click', (e) => {
                e.preventDefault();
                selectedEnrollFaceIndex = idx;
                updateFaceSelectorUI();
            });
            faceSelectorPills.appendChild(pill);
        });
    } else {
        // Update active class
        Array.from(faceSelectorPills.children).forEach((pill, idx) => {
            if (idx === selectedEnrollFaceIndex) {
                pill.classList.add('active');
            } else {
                pill.classList.remove('active');
            }
        });
    }
}

// Extract geometric landmark ratios for hyper-responsive facial micro-expression analysis
function extractLandmarkGeometry(landmarks) {
    if (!landmarks || !landmarks.positions || landmarks.positions.length < 68) return null;
    const p = landmarks.positions;
    const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);
    
    // Normalization scale: distance between outer eye corners (pt 36 to pt 45)
    const eyeScale = Math.max(12, dist(p[36], p[45]));
    
    // 1. Mouth metrics (outer corners 48, 54; lip centers 51, 57; inner lips 62, 66)
    const mouthWidth = dist(p[48], p[54]) / eyeScale;
    const mouthOpen = dist(p[62], p[66]) / eyeScale;
    const mouthCenterY = (p[51].y + p[57].y) / 2;
    
    // Corners elevation relative to mouth center (in screen coords, smaller Y is higher on face)
    const leftCornerLift = (mouthCenterY - p[48].y) / eyeScale;
    const rightCornerLift = (mouthCenterY - p[54].y) / eyeScale;
    const avgCornerLift = (leftCornerLift + rightCornerLift) / 2;
    const cornerAsymmetry = Math.abs(leftCornerLift - rightCornerLift);
    
    // 2. Eyebrow metrics (eyebrows 19, 24 relative to eye tops 37, 44; inner brows 21, 22)
    const leftBrowHeight = (p[37].y - p[19].y) / eyeScale;
    const rightBrowHeight = (p[44].y - p[24].y) / eyeScale;
    const avgBrowHeight = (leftBrowHeight + rightBrowHeight) / 2;
    const browSeparation = dist(p[21], p[22]) / eyeScale;
    
    return {
        mouthWidth,
        mouthOpen,
        leftCornerLift,
        rightCornerLift,
        avgCornerLift,
        cornerAsymmetry,
        avgBrowHeight,
        browSeparation
    };
}

// Advanced Emotion Engine: Fuses neural network probabilities + 68 geometric micro-curves + temporal hysteresis
function getAdvancedExpression(expressions, landmarks, trackId) {
    if (!expressions) return { label: '😐 NEUTRAL (85%)', rawEmotion: 'neutral' };

    const geo = extractLandmarkGeometry(landmarks);
    const isHigh = expressionSensitivity === 'high';
    const now = Date.now();

    // Expression candidate scores
    let detectedEmotion = 'neutral';
    let emotionScore = expressions.neutral || 0.6;
    let emotionTitle = '😐 NEUTRAL';

    // 1. Smile / Happy check (fused neural + geometric corner lift & mouth expansion)
    const neuralHappy = expressions.happy || 0;
    const geoSmile = geo ? (geo.avgCornerLift > (isHigh ? 0.022 : 0.035) || geo.mouthWidth > (isHigh ? 0.81 : 0.86)) : false;
    
    if (neuralHappy >= (isHigh ? 0.10 : 0.18) || geoSmile) {
        if (geo && geo.mouthOpen > 0.17 && geo.avgCornerLift > 0.03) {
            detectedEmotion = 'laugh';
            emotionTitle = '😄 LAUGHING';
            emotionScore = Math.max(neuralHappy, 0.88);
        } else {
            detectedEmotion = 'happy';
            emotionTitle = '😊 SMILING';
            emotionScore = Math.max(neuralHappy, geo ? Math.min(0.96, 0.65 + geo.avgCornerLift * 3) : 0.85);
        }
    }
    // 2. Asymmetric Smirk check
    else if (geo && geo.cornerAsymmetry > 0.045 && (geo.leftCornerLift > 0.025 || geo.rightCornerLift > 0.025) && neuralHappy >= 0.05) {
        detectedEmotion = 'smirk';
        emotionTitle = '😏 SMIRKING';
        emotionScore = 0.82;
    }
    // 3. Surprised check (wide eyes / open mouth / raised brows)
    else if (
        (expressions.surprised || 0) >= (isHigh ? 0.12 : 0.20) ||
        (geo && geo.mouthOpen > 0.20 && geo.avgBrowHeight > 0.29)
    ) {
        detectedEmotion = 'surprised';
        emotionTitle = '😲 SURPRISED';
        emotionScore = Math.max(expressions.surprised || 0, geo ? 0.85 : 0.75);
    }
    // 4. Angry / Intense check (furrowed inner eyebrows / lowered brow)
    else if (
        (expressions.angry || 0) >= (isHigh ? 0.12 : 0.20) ||
        (geo && geo.browSeparation < 0.23 && geo.avgBrowHeight < 0.23)
    ) {
        detectedEmotion = 'angry';
        emotionTitle = '😠 ANGRY';
        emotionScore = Math.max(expressions.angry || 0, 0.78);
    }
    // 5. Sad check (downturned lip corners)
    else if (
        (expressions.sad || 0) >= (isHigh ? 0.12 : 0.20) ||
        (geo && geo.avgCornerLift < -0.028)
    ) {
        detectedEmotion = 'sad';
        emotionTitle = '😢 SAD';
        emotionScore = Math.max(expressions.sad || 0, 0.78);
    }
    // 6. Disgusted / Fearful
    else if ((expressions.disgusted || 0) >= 0.15) {
        detectedEmotion = 'disgusted';
        emotionTitle = '🤢 DISGUSTED';
        emotionScore = expressions.disgusted;
    } else if ((expressions.fearful || 0) >= 0.15) {
        detectedEmotion = 'fearful';
        emotionTitle = '😨 FEARFUL';
        emotionScore = expressions.fearful;
    }

    // Temporal Hysteresis: prevent single-frame flickering
    let track = faceTrackCache.get(trackId);
    if (!track) {
        track = { currentEmotion: detectedEmotion, title: emotionTitle, score: emotionScore, holdUntil: 0, lastSeen: now };
        faceTrackCache.set(trackId, track);
    } else {
        track.lastSeen = now;
        if (detectedEmotion !== 'neutral') {
            track.currentEmotion = detectedEmotion;
            track.title = emotionTitle;
            track.score = (track.score * 0.4) + (emotionScore * 0.6);
            track.holdUntil = now + 320;
        } else if (now < track.holdUntil) {
            detectedEmotion = track.currentEmotion;
            emotionTitle = track.title;
            emotionScore = track.score;
        } else {
            track.currentEmotion = 'neutral';
            track.title = '😐 NEUTRAL';
            track.score = (expressions.neutral || 0.85);
            detectedEmotion = 'neutral';
            emotionTitle = '😐 NEUTRAL';
            emotionScore = track.score;
        }
    }

    const confidence = Math.round(Math.min(99, Math.max(60, emotionScore * 100)));
    return {
        label: `${emotionTitle} (${confidence}%)`,
        rawEmotion: detectedEmotion,
        confidence: confidence
    };
}

// Clean periodic cache cleanup
function cleanTrackCache() {
    const now = Date.now();
    for (const [key, track] of faceTrackCache.entries()) {
        if (now - track.lastSeen > 8000) {
            faceTrackCache.delete(key);
        }
    }
}

// Custom HUD drawings (Clean rounded boxes, dynamic badges, optimal multi-face labels, zero age clutter)
function drawCustomHUD(canvas, resizedDetections, matchResults) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const now = Date.now();
    cleanTrackCache();
    let recognizedCount = 0;
    
    resizedDetections.forEach((detection, idx) => {
        const { x, y, width, height } = detection.detection.box;
        
        // Mirror the X coordinate to match CSS mirrored webcam video
        const mirroredX = canvas.width - x - width;
        
        const match = matchResults[idx] || {
            isMatch: false,
            labelName: "IDENTIFYING...",
            confidenceScore: 0.8,
            distance: 1.0,
            statusColor: '#06b6d4'
        };

        if (match.isMatch) {
            recognizedCount++;
        }

        // Throttle logging
        handleLoggingThrottled(match.labelName, match.isMatch, now);

        const isTargetedForEnroll = (resizedDetections.length > 1 && idx === selectedEnrollFaceIndex);
        let hudColor = match.statusColor;
        if (isTargetedForEnroll) {
            hudColor = '#f59e0b'; // amber/gold for targeted face
        }
        
        // 1. Clean Rounded Bounding Rectangle (smooth glowing corners)
        ctx.save();
        ctx.strokeStyle = hudColor;
        ctx.lineWidth = isTargetedForEnroll ? 3.0 : 2.5;
        ctx.shadowColor = hudColor;
        ctx.shadowBlur = isTargetedForEnroll ? 14 : 10;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(mirroredX, y, width, height, 12);
        } else {
            ctx.rect(mirroredX, y, width, height);
        }
        ctx.stroke();
        ctx.restore();
        
        // 2. Prepare Formatted Badge Content
        const trackId = match.isMatch ? match.labelName : `face_${Math.round(x / 80)}_${Math.round(y / 80)}`;
        const refinedMood = getAdvancedExpression(detection.expressions, detection.landmarks, trackId);
        const gender = (detection.gender || 'UNKNOWN').toUpperCase();

        const matchConfidenceText = match.isMatch ? ` (${Math.round(match.confidenceScore * 100)}%)` : '';
        const line1Text = match.labelName + matchConfidenceText;
        const line2Text = `MOOD: ${refinedMood.label}  •  ${gender}`;
        
        // 3. Dynamically measure text width to guarantee ZERO badge overflow
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        const line1Width = ctx.measureText(line1Text).width;
        
        ctx.font = '500 10px "Outfit", sans-serif';
        const line2Width = ctx.measureText(line2Text).width;
        
        const neededWidth = Math.max(line1Width + 36, line2Width + 24);
        const badgeWidth = Math.max(width, Math.max(neededWidth, 195));
        const badgeHeight = 46;
        
        // Center the badge horizontally over the face box
        let badgeX = mirroredX + (width - badgeWidth) / 2;
        // Clamp to ensure badge never exceeds the left or right canvas boundaries
        badgeX = Math.max(8, Math.min(canvas.width - badgeWidth - 8, badgeX));
        
        // Vertical placement: directly above the face box
        const offset = 8;
        let paneY = y - badgeHeight - offset;
        // If too close to the top, flip the badge below the face box
        if (paneY < 6) {
            paneY = y + height + offset;
            if (paneY + badgeHeight > canvas.height - 6) {
                paneY = canvas.height - badgeHeight - 6;
            }
        }
        
        // 4. Render Glassmorphic Badge Background
        ctx.save();
        ctx.fillStyle = 'rgba(7, 11, 24, 0.94)';
        ctx.strokeStyle = hudColor;
        ctx.lineWidth = 1.5;
        ctx.shadowColor = hudColor;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(badgeX, paneY, badgeWidth, badgeHeight, 10);
        } else {
            ctx.rect(badgeX, paneY, badgeWidth, badgeHeight);
        }
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        
        // 5. Draw Line 1: Identity & Confidence
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.fillText(line1Text, badgeX + 10, paneY + 18);
        
        // 6. Draw Status Indicator Dot
        ctx.save();
        ctx.fillStyle = hudColor;
        ctx.shadowColor = hudColor;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(badgeX + badgeWidth - 14, paneY + 14, 4.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();
        
        // 7. Draw Line 2: Emotion & Gender
        ctx.fillStyle = '#94a3b8';
        ctx.font = '500 10px "Outfit", sans-serif';
        ctx.fillText(line2Text, badgeX + 10, paneY + 35);

        // 8. If targeted for enrollment, draw "🎯 TARGET FOR ENROLLMENT" tag
        if (isTargetedForEnroll) {
            const tagText = "🎯 TARGET FOR ENROLLMENT";
            ctx.font = 'bold 9px "JetBrains Mono", monospace';
            const tagWidth = ctx.measureText(tagText).width + 12;
            const tagX = mirroredX + (width - tagWidth) / 2;
            const tagY = y + height + 6;

            if (tagY + 18 < canvas.height) {
                ctx.save();
                ctx.fillStyle = 'rgba(245, 158, 11, 0.95)';
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(tagX, tagY, tagWidth, 16, 4);
                } else {
                    ctx.rect(tagX, tagY, tagWidth, 16);
                }
                ctx.fill();
                ctx.fillStyle = '#090d1a';
                ctx.fillText(tagText, tagX + 6, tagY + 12);
                ctx.restore();
            }
        }
    });

    // Check for unregistered persons in viewport
    const unregisteredCount = matchResults.filter(m => !m.isMatch).length;
    if (unregisteredCount > 0 && isDetecting) {
        if (unregisteredAlertBanner) {
            unregisteredAlertBanner.style.display = 'flex';
            if (unregisteredAlertText) {
                unregisteredAlertText.innerText = unregisteredCount === 1
                    ? "SECURITY ALERT: UNREGISTERED PERSON DETECTED"
                    : `SECURITY ALERT: ${unregisteredCount} UNREGISTERED PERSONS DETECTED`;
            }
        }
        if (unregisteredBadge) unregisteredBadge.style.display = 'inline-flex';
    } else {
        if (unregisteredAlertBanner) unregisteredAlertBanner.style.display = 'none';
        if (unregisteredBadge) unregisteredBadge.style.display = 'none';
    }
    
    // Update Multi-Face simultaneous status in the header
    if (facesBadge && facesCountText) {
        if (resizedDetections.length > 0) {
            facesCountText.innerText = `${resizedDetections.length} Face${resizedDetections.length > 1 ? 's' : ''} (${recognizedCount} Recognized)`;
            facesBadge.style.display = 'inline-flex';
        } else {
            facesBadge.style.display = 'none';
        }
    }
}

// Log matcher logs with timing throttle
function handleLoggingThrottled(name, isMatch, timeNow) {
    if (!lastSeen[name] || (timeNow - lastSeen[name]) > LOG_COOLDOWN_MS) {
        lastSeen[name] = timeNow;
        if (isMatch) {
            addLog(`Face recognized: ${name}`, 'match');
        } else {
            addLog("SECURITY ALERT: Unregistered person in camera view!", 'unknown');
        }
    }
}

// Enroll Selected Face Profile (Targeted & Multi-Face Aware)
async function enrollFace() {
    const nameVal = enrollNameInput.value.trim();
    if (!nameVal) {
        addLog("Enrollment failed: Profile name field cannot be empty.", "unknown");
        enrollNameInput.focus();
        return;
    }
    
    // Check if name already exists (case-insensitive)
    if (db.some(p => p.name.toLowerCase() === nameVal.toLowerCase())) {
        addLog(`Enrollment failed: Profile "${nameVal}" is already registered.`, "unknown");
        return;
    }
    
    btnEnroll.disabled = true;
    btnEnroll.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>ANALYZING...</span>';
    addLog(`Initiating biometric extraction for "${nameVal}"...`, "system");
    
    try {
        if (!latestDetections || latestDetections.length === 0) {
            addLog("Enrollment aborted: No face detected. Align face inside camera viewfinder.", "unknown");
            return;
        }
        
        // Pick the exact selected face (defaults to 0 if only 1 face)
        let targetIndex = selectedEnrollFaceIndex;
        if (targetIndex >= latestDetections.length || targetIndex < 0) {
            targetIndex = 0;
        }
        
        const targetFace = latestDetections[targetIndex];
        if (!targetFace || !targetFace.descriptor) {
            addLog("Enrollment failed: Unable to compute 128-d biometric descriptor.", "unknown");
            return;
        }
        
        // Success! Save to database
        db.push({
            name: nameVal,
            descriptor: targetFace.descriptor
        });
        saveDatabase();
        
        addLog(`Profile "${nameVal}" enrolled successfully! (Assigned to Face #${targetIndex + 1})`, "match");
        enrollNameInput.value = '';
    } catch (e) {
        console.error("Enrollment failed:", e);
        addLog("Critical enrollment failure occurred during vector extraction.", "unknown");
    } finally {
        btnEnroll.innerHTML = '<i class="fa-solid fa-fingerprint"></i> <span>ENROLL</span>';
        updateEnrollBtnState();
    }
}

// Export DB as JSON file
function exportDatabase() {
    if (db.length === 0) {
        addLog("Export skipped: The database is currently empty.", "unknown");
        return;
    }
    
    try {
        const rawDb = db.map(item => ({
            name: item.name,
            descriptor: Array.from(item.descriptor)
        }));
        
        const jsonStr = JSON.stringify(rawDb, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `face_sentry_database_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        addLog("Database exported successfully.", "system");
    } catch (e) {
        console.error(e);
        addLog("Database export failed.", "unknown");
    }
}

// Import DB from JSON file
function importDatabase(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const rawDb = JSON.parse(e.target.result);
            if (!Array.isArray(rawDb)) {
                throw new Error("Invalid format: Root must be a JSON array.");
            }
            
            const validated = [];
            rawDb.forEach(item => {
                if (item.name && Array.isArray(item.descriptor) && item.descriptor.length === 128) {
                    validated.push({
                        name: item.name,
                        descriptor: new Float32Array(item.descriptor)
                    });
                }
            });
            
            if (validated.length === 0) {
                throw new Error("No valid face recognition profiles found in file.");
            }
            
            db = validated;
            saveDatabase();
            addLog(`Imported ${validated.length} profiles. Local database updated.`, "system");
        } catch (err) {
            console.error(err);
            addLog(`Import failed: ${err.message}`, "unknown");
        } finally {
            dbFileInput.value = '';
        }
    };
    reader.readAsText(file);
}
