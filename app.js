// AI Sentry - Face Recognition Engine
// Powered by face-api.js (vladmandic fork)

// State variables
let db = [];
let stream = null;
let isDetecting = false;
let detectorType = 'tiny'; // 'tiny' or 'ssd'
let minConfidence = 0.4;
let totalScanCount = 0;
let lastSeen = {}; // Tracker to prevent spamming logs
const LOG_COOLDOWN_MS = 5000; // Log matches once every 5 seconds
let latestDetections = []; // Stores current frame detections for immediate enrollment
let expressionSensitivity = 'high'; // 'high' (landmark-boosted) or 'balanced'

// Multi-face track cache for expression smoothing and hysteresis
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
const scanIcon = document.getElementById('scan-icon');
const videoIcon = document.getElementById('video-icon');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const cameraSelect = document.getElementById('camera-select');
const btnRefreshCameras = document.getElementById('btn-refresh-cameras');
let selectedDeviceId = localStorage.getItem('face_sentry_camera_id') || '';
const facesBadge = document.getElementById('faces-badge');
const facesCountText = document.getElementById('faces-count-text');
const btnSensHigh = document.getElementById('btn-sens-high');
const btnSensStd = document.getElementById('btn-sens-std');
const expressionModeVal = document.getElementById('expression-mode-val');
const enrollNameInput = document.getElementById('enroll-name');
const btnEnroll = document.getElementById('btn-enroll');
const dbGrid = document.getElementById('db-grid');
const dbCountEl = document.getElementById('db-count');
const btnExportDb = document.getElementById('btn-export-db');
const btnImportDb = document.getElementById('btn-import-db');
const dbFileInput = document.getElementById('db-file-input');
const btnClearLogs = document.getElementById('btn-clear-logs');
const logContainer = document.getElementById('log-container');
const selTiny = document.getElementById('sel-tiny');
const selSsd = document.getElementById('sel-ssd');
const fpsVal = document.getElementById('fps-val');
const statModelType = document.getElementById('stat-model-type');
const statMinConf = document.getElementById('stat-min-conf');
const statScanCount = document.getElementById('stat-scan-count');

const sysStatusDot = document.getElementById('sys-status-dot');
const sysStatusText = document.getElementById('sys-status-text');
const modelStatusDot = document.getElementById('model-status-dot');
const modelStatusText = document.getElementById('model-status-text');
const cameraStatusDot = document.getElementById('camera-status-dot');
const cameraStatusText = document.getElementById('camera-status-text');

// Init application
document.addEventListener('DOMContentLoaded', async () => {
    loadDatabase();
    statMinConf.innerText = Math.round(minConfidence * 100) + '%';
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
            addLog(`Loaded ${db.length} profiles from local database.`, 'system');
        } else {
            addLog("Database is empty. Enrolling first profile recommended.", 'system');
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
        dbGrid.innerHTML = '<div class="empty-state">No enrolled profiles</div>';
        return;
    }

    db.forEach((profile, index) => {
        const item = document.createElement('div');
        item.className = 'db-item';
        
        // Generate initials
        const parts = profile.name.split(' ');
        const initials = parts.map(p => p[0]).join('').substring(0, 2).toUpperCase();
        
        item.innerHTML = `
            <div class="db-avatar">${initials}</div>
            <div class="db-name" title="${profile.name}">${profile.name}</div>
            <div class="delete-btn" title="Delete Profile" data-index="${index}">&times;</div>
        `;
        
        item.querySelector('.delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteProfile(index);
        });
        
        dbGrid.appendChild(item);
    });
}

// Delete Profile
function deleteProfile(index) {
    const name = db[index].name;
    db.splice(index, 1);
    saveDatabase();
    addLog(`Deleted profile: ${name}`, 'system');
    updateEnrollBtnState();
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
    
    // Cap logs at 50 to keep things fast
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
        
        addLog("Loading AI Facial Landmark network...", "system");
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        
        addLog("Loading AI Face Recognition vectorizer...", "system");
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        
        addLog("Loading AI Mood & Demographics networks...", "system");
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);
        
        modelStatusDot.className = 'status-dot active';
        modelStatusText.innerText = 'READY';
        addLog("AI Neural Networks loaded successfully. System operational.", "system");
        
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

// Event Listeners Setup
function setupEventListeners() {
    btnToggleCamera.addEventListener('click', toggleCamera);

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

    // Model Selector
    selTiny.addEventListener('click', () => {
        if (detectorType === 'tiny') return;
        detectorType = 'tiny';
        minConfidence = 0.4;
        selTiny.classList.add('active');
        selSsd.classList.remove('active');
        statModelType.innerText = 'TINY';
        statMinConf.innerText = '40%';
        addLog("Detector set to Tiny Face Detector (Low Latency / CPU efficient)", "system");
    });

    selSsd.addEventListener('click', () => {
        if (detectorType === 'ssd') return;
        detectorType = 'ssd';
        minConfidence = 0.4;
        selSsd.classList.add('active');
        selTiny.classList.remove('active');
        statModelType.innerText = 'SSD';
        statMinConf.innerText = '40%';
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

        // Get actual active track settings/label
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
        addLog("Camera stream established. Clean facial monitoring online.", "system");
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
        faceTrackCache.clear();
        
        updateEnrollBtnState();
        fpsVal.innerText = '00';
        addLog("Facial monitoring feed terminated.", "system");
    }
}

// Recognition Engine & Canvas Drawing Loop
async function onPlay() {
    if (!stream || webcam.paused || webcam.ended) return;
    
    // Check if video dimensions are fully resolved
    if (webcam.videoWidth === 0 || webcam.videoHeight === 0) {
        setTimeout(onPlay, 100);
        return;
    }
    
    isDetecting = true;
    
    // FPS tracking variables
    let lastTime = performance.now();
    let frameCount = 0;
    let fps = 0;
    
    // Create face matcher from current DB
    let faceMatcher = getFaceMatcher();

    // Loop
    async function detectionLoop() {
        if (!isDetecting || webcam.paused || webcam.ended) return;
        
        // Periodically refresh matcher in case DB changes
        if (frameCount % 30 === 0) {
            faceMatcher = getFaceMatcher();
        }
        
        // Select detector options (416 input size provides high resolution for multiple simultaneous faces)
        const options = detectorType === 'tiny'
            ? new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: minConfidence })
            : new faceapi.SsdMobilenetv1Options({ minConfidence: minConfidence });
            
        try {
            // Match canvas overlay size to video bounds
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
                
            // Store latest detections for enrollment
            latestDetections = detections;
                
            // Update scanning counts
            if (detections.length > 0) {
                totalScanCount += detections.length;
                statScanCount.innerText = totalScanCount;
            }
            
            // Resize outputs to overlay dimensions
            const displaySize = { width: webcam.videoWidth, height: webcam.videoHeight };
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            
            // Draw custom premium overlays
            drawCustomHUD(overlay, resizedDetections, faceMatcher);
            
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

// Convert DB array to LabeledFaceDescriptors
function getFaceMatcher() {
    if (db.length === 0) return null;
    
    const labeledDescriptors = db.map(profile => {
        return new faceapi.LabeledFaceDescriptors(profile.name, [profile.descriptor]);
    });
    
    // Threshold Euclidean distance: 0.55 (smaller is stricter)
    return new faceapi.FaceMatcher(labeledDescriptors, 0.55);
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
            // Active emotion lock: hold for at least 320ms so it displays smoothly
            track.currentEmotion = detectedEmotion;
            track.title = emotionTitle;
            track.score = (track.score * 0.4) + (emotionScore * 0.6);
            track.holdUntil = now + 320;
        } else if (now < track.holdUntil) {
            // Keep holding the active emotion until hold timer elapses
            detectedEmotion = track.currentEmotion;
            emotionTitle = track.title;
            emotionScore = track.score;
        } else {
            // Decayed back to neutral
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

// Custom clean HUD drawings (No scanning laser, no dots, no brackets, NO AGE DISPLAY)
function drawCustomHUD(canvas, resizedDetections, faceMatcher) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const now = Date.now();
    cleanTrackCache();
    let recognizedCount = 0;
    
    resizedDetections.forEach((detection, idx) => {
        const { x, y, width, height } = detection.detection.box;
        
        // Mirror the X coordinate to match the CSS mirrored webcam video
        const mirroredX = canvas.width - x - width;
        
        let labelName = "IDENTIFYING...";
        let isMatch = false;
        let confidenceScore = detection.detection.score;
        
        // Match each face independently across all enrolled profiles
        if (faceMatcher) {
            const match = faceMatcher.findBestMatch(detection.descriptor);
            if (match.label !== 'unknown') {
                labelName = match.label.toUpperCase();
                isMatch = true;
                confidenceScore = Math.max(0, Math.min(1, 1 - match.distance));
                recognizedCount++;
            } else {
                labelName = "SUSPECT UNKNOWN";
            }
        } else {
            labelName = "UNREGISTERED TARGET";
        }
        
        // Log entries throttling
        handleLoggingThrottled(labelName, isMatch, now);
        
        // Theme color: Green for enrolled match, Rose/Red for unknown, Cyan for searching
        let hudColor = '#06b6d4'; // neon cyan
        if (isMatch) {
            hudColor = '#10b981'; // neon emerald
        } else if (labelName === "SUSPECT UNKNOWN") {
            hudColor = '#f43f5e'; // neon rose
        }
        
        // 1. Clean Rounded Bounding Rectangle (smooth corners, glowing outline)
        ctx.save();
        ctx.strokeStyle = hudColor;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = hudColor;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(mirroredX, y, width, height, 12);
        } else {
            ctx.rect(mirroredX, y, width, height);
        }
        ctx.stroke();
        ctx.restore();
        
        // 2. Prepare Formatted Badge Content
        const trackId = isMatch ? labelName : `face_${Math.round(x / 80)}_${Math.round(y / 80)}`;
        const refinedMood = getAdvancedExpression(detection.expressions, detection.landmarks, trackId);
        const gender = (detection.gender || 'UNKNOWN').toUpperCase();

        const matchConfidenceText = isMatch ? ` (${Math.round(confidenceScore * 100)}%)` : '';
        const line1Text = labelName + matchConfidenceText;
        const line2Text = `MOOD: ${refinedMood.label}  •  ${gender}`;
        
        // 3. Dynamically measure text width to prevent ANY overflow
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        const line1Width = ctx.measureText(line1Text).width;
        
        ctx.font = '500 10px "Outfit", sans-serif';
        const line2Width = ctx.measureText(line2Text).width;
        
        // Ensure badge is wide enough for the text + comfortable padding + indicator dot
        const neededWidth = Math.max(line1Width + 34, line2Width + 24);
        const badgeWidth = Math.max(width, Math.max(neededWidth, 190));
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
            // Prevent going below the bottom canvas boundary
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
        
        // 5. Draw Line 1: Profile Identity & Confidence
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.fillText(line1Text, badgeX + 10, paneY + 18);
        
        // 6. Draw Status Indicator Dot (positioned cleanly at the far right edge of the badge)
        ctx.save();
        ctx.fillStyle = hudColor;
        ctx.shadowColor = hudColor;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(badgeX + badgeWidth - 14, paneY + 14, 4.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();
        
        // 7. Draw Line 2: Emotion & Gender (clean, spacious, zero clipping)
        ctx.fillStyle = '#94a3b8';
        ctx.font = '500 10px "Outfit", sans-serif';
        ctx.fillText(line2Text, badgeX + 10, paneY + 35);
    });
    
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
        } else if (name === "SUSPECT UNKNOWN") {
            addLog("Alert: Unregistered face in viewport.", 'unknown');
        } else if (name === "UNREGISTERED TARGET") {
            addLog("Detection: Unregistered face detected.", 'unknown');
        }
    }
}

// Enroll Current Target Face (smart multi-face aware)
async function enrollFace() {
    const nameVal = enrollNameInput.value.trim();
    if (!nameVal) {
        addLog("Enrollment failed: Name field cannot be empty.", "unknown");
        enrollNameInput.focus();
        return;
    }
    
    // Check if name already exists
    if (db.some(p => p.name.toLowerCase() === nameVal.toLowerCase())) {
        addLog(`Enrollment failed: Profile name "${nameVal}" is already registered.`, "unknown");
        return;
    }
    
    btnEnroll.disabled = true;
    btnEnroll.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>ANALYZING...</span>';
    addLog(`Initiating facial biometric extraction for "${nameVal}"...`, "system");
    
    try {
        if (!latestDetections || latestDetections.length === 0) {
            addLog("Enrollment aborted: Face not detected. Align face inside viewfinder.", "unknown");
            return;
        }
        
        let targetFace = null;
        if (latestDetections.length === 1) {
            targetFace = latestDetections[0];
        } else {
            // Smart multi-face enrollment: if some faces are already registered, enroll the unregistered face!
            const faceMatcher = getFaceMatcher();
            if (faceMatcher) {
                const unregistered = latestDetections.filter(d => {
                    const match = faceMatcher.findBestMatch(d.descriptor);
                    return match.label === 'unknown';
                });
                if (unregistered.length === 1) {
                    targetFace = unregistered[0];
                    addLog("Multiple faces in view: Automatically enrolling the unregistered face.", "system");
                } else if (unregistered.length > 1) {
                    unregistered.sort((a, b) => (b.detection.box.width * b.detection.box.height) - (a.detection.box.width * a.detection.box.height));
                    targetFace = unregistered[0];
                    addLog("Multiple unregistered faces detected. Enrolling the primary face in focus.", "system");
                }
            }
            if (!targetFace) {
                // Pick the largest / most centered face
                const sorted = [...latestDetections].sort((a, b) => (b.detection.box.width * b.detection.box.height) - (a.detection.box.width * a.detection.box.height));
                targetFace = sorted[0];
            }
        }
        
        // Success! Save to database
        db.push({
            name: nameVal,
            descriptor: targetFace.descriptor
        });
        saveDatabase();
        addLog(`Face enrolled successfully: Profile "${nameVal}" established.`, "match");
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
            
            // Validate schema roughly
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
            
            // Merge or replace options (we will replace for simplicity)
            db = validated;
            saveDatabase();
            addLog(`Imported ${validated.length} profiles. Local database updated.`, "system");
        } catch (err) {
            console.error(err);
            addLog(`Import failed: ${err.message}`, "unknown");
        } finally {
            dbFileInput.value = ''; // Reset input
        }
    };
    reader.readAsText(file);
}
