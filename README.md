# AI Sentry - Face Recognition Engine

A futuristic, high-performance, real-time Artificial Intelligence Face Recognition dashboard that runs entirely inside your web browser. 

Using neural networks running client-side, the app detects faces, plots 68 facial landmarks, estimates age and gender, predicts emotional states, and recognizes registered profiles using high-accuracy face descriptor embeddings.

---

## Key Features

*   **Real-time Video Processing:** Detects and tracks faces at up to 60 FPS using your laptop's built-in webcam.
*   **Futuristic Cyber HUD Overlay:** Beautiful custom Canvas-based visualization highlighting faces with glowing neon crosshairs, landmarks, and indicators based on identification status (green for matches, red for unknowns, cyan for searching).
*   **Mood & Demographic Analytics:** Provides age estimation, gender identification, and real-time dominant emotion analysis (emoji-coded).
*   **Local Database Profiles:** Enroll profiles directly from the feed. Biometric descriptors are saved directly to your browser's secure `localStorage` (no image files are uploaded or stored, keeping it private and GDPR compliant).
*   **Database Import/Export:** Save your database as a portable `.json` file to back up or share enrolled faces.
*   **Dual Detector Support:** Switch between the ultra-fast Tiny Face Detector (best for standard laptops) and the high-accuracy SSD MobileNet V1 model.
*   **Zero-Dependency Server:** Starts a local HTTP server using native Windows PowerShell, avoiding the need for Node.js, Python, or Web server installations.

---

## Quick Start Instructions

Because browsers restrict camera access (`getUserMedia`) to secure contexts (`https://` or `localhost`), loading the page directly as a file (`file://index.html`) will prevent the camera from opening. Follow these simple steps to run the server:

1.  **Open PowerShell** in the project directory (`c:\Users\ganes\AI`).
2.  **Start the Local Server** by running:
    ```powershell
    .\serve.ps1
    ```
3.  **Automatic Launch:** The script will automatically open your default browser to:
    `http://localhost:8000`
4.  **Allow Camera Access:** Click **Allow** when your browser requests access to your webcam.
5.  **Model Loading:** On first start, the app will download the neural network weights from jsDelivr CDN (~7 MB total). It will take a few seconds. The top status bar will change to **MODELS: READY**.
6.  **Start Monitoring:** Click **START MONITORING** to activate the webcam feed!

---

## How to Register & Recognize Profiles

1.  **Enrollment:**
    *   Align yourself in the camera viewport.
    *   In the **ENROLL NEW FACE** sidebar card, enter a name (e.g., "Alice").
    *   Click **ENROLL**. 
    *   The AI will extract your 128-dimension face vector and add it to the **ENROLLED PROFILES** database.
2.  **Recognition:**
    *   Look back at the camera.
    *   The HUD crosshair will change to **green** and display your name (e.g., "ALICE") alongside the match confidence rating.
    *   Unregistered faces will show a **red** frame labeled `SUSPECT UNKNOWN`.
3.  **Logging:**
    *   The bottom **LOG STREAM** maintains a real-time running log of matches, unknown target warnings, and system events.

---

## Technical Architecture

*   **Front-End Core:** HTML5, modern vanilla CSS, and Vanilla ES6 JavaScript.
*   **AI Engine:** `face-api.js` (maintained fork by Vladimir Mandic) powered by TensorFlow.js (compiled automatically to WebGL/CPU).
*   **Biometrics:** Face embeddings are 128-float coordinate arrays. Distance is computed using Euclidean Distance metrics.
*   **Local Server:** Native C# `System.Net.HttpListener` wrapped in a PowerShell loop.
