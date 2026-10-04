// =============================================================
// NAV TOGGLE (mobile hamburger menu) - runs on every page
// =============================================================

const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");

if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
        navLinks.classList.toggle("open");
    });
}


// =============================================================
// HOME PAGE
// =============================================================

const startButton = document.getElementById("startButton");

if (startButton) {
    startButton.addEventListener("click", function () {
        window.location.href = "analysis.html";
    });
}


// =============================================================
// REAL-TIME ANALYSIS PAGE
// =============================================================

const startAnalysis = document.getElementById("startAnalysis");
const stopAnalysis = document.getElementById("stopAnalysis");
const timerEl = document.getElementById("timer");

if (startAnalysis && stopAnalysis) {

    // ---- DOM refs ----
    const micBanner = document.getElementById("micBanner");
    const micBannerText = document.getElementById("micBannerText");
    const speechRateValue = document.getElementById("speechRateValue");
    const speechRateStatus = document.getElementById("speechRateStatus");
    const pitchValue = document.getElementById("pitchValue");
    const pitchStatus = document.getElementById("pitchStatus");
    const pauseValue = document.getElementById("pauseValue");
    const pauseStatus = document.getElementById("pauseStatus");
    const fillerValue = document.getElementById("fillerValue");
    const fillerStatus = document.getElementById("fillerStatus");
    const qualityStatus = document.getElementById("qualityStatus");
    const liveTranscript = document.getElementById("liveTranscript");
    const canvas = document.getElementById("pitchCanvas");
    const canvasCtx = canvas ? canvas.getContext("2d") : null;

    const FILLER_WORDS = new Set([
        "um", "umm", "uh", "uhh", "erm", "er", "ah", "like", "youknow"
    ]);

    // ---- session state ----
    let seconds = 0;
    let timerInterval = null;
    let audioContext = null;
    let analyser = null;
    let micStream = null;
    let animationFrameId = null;
    let recognition = null;
    let recognitionSupported = false;

    let pauseCount = 0;
    let isSilent = false;
    let silenceStartTime = 0;
    const SILENCE_RMS_THRESHOLD = 0.02;
    const SILENCE_MIN_DURATION_MS = 600;

    let pitchHistory = [];       // all confident pitch readings, for mean/std at the end
    let pitchDrawHistory = [];   // recent readings, for the live canvas line

    let totalWords = 0;
    let fillerCount = 0;
    let sessionStartTime = 0;

    // Recording of the raw mic audio, sent to /api/analyze on Stop so the
    // REAL trained Conformer model (not just the JS heuristic below) can
    // judge quality. mediaRecorder stays null (and we silently fall back
    // to the heuristic) if MediaRecorder isn't supported in this browser.
    let mediaRecorder = null;
    let recordedChunks = [];

    // ---------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------

    function setStatusClass(el, level) {
        el.classList.remove("good", "average", "poor");
        if (level) el.classList.add(level);
    }

    function showBanner(type, message) {
        micBanner.classList.remove("info", "error");
        micBanner.classList.add("show", type);
        micBannerText.textContent = message;
    }

    function hideBanner() {
        micBanner.classList.remove("show");
    }

    function formatTimer(totalSeconds) {
        const h = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
        const m = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
        const s = String(totalSeconds % 60).padStart(2, "0");
        return h + ":" + m + ":" + s;
    }

    // Basic autocorrelation-based pitch detector.
    // Returns { freq, rms }. freq is -1 if no confident pitch found (silence/noise).
    function autoCorrelate(buffer, sampleRate) {
        const SIZE = buffer.length;
        let rms = 0;
        for (let i = 0; i < SIZE; i++) {
            rms += buffer[i] * buffer[i];
        }
        rms = Math.sqrt(rms / SIZE);
        if (rms < SILENCE_RMS_THRESHOLD) return { freq: -1, rms };

        let r1 = 0, r2 = SIZE - 1;
        const threshold = 0.2;
        for (let i = 0; i < SIZE / 2; i++) {
            if (Math.abs(buffer[i]) < threshold) { r1 = i; break; }
        }
        for (let i = 1; i < SIZE / 2; i++) {
            if (Math.abs(buffer[SIZE - i]) < threshold) { r2 = SIZE - i; break; }
        }

        const trimmed = buffer.slice(r1, r2);
        const n = trimmed.length;
        const c = new Array(n).fill(0);
        for (let lag = 0; lag < n; lag++) {
            for (let i = 0; i < n - lag; i++) {
                c[lag] += trimmed[i] * trimmed[i + lag];
            }
        }

        let d = 0;
        while (d < n - 1 && c[d] > c[d + 1]) d++;

        let maxVal = -1, maxPos = -1;
        for (let i = d; i < n; i++) {
            if (c[i] > maxVal) { maxVal = c[i]; maxPos = i; }
        }

        const T0 = maxPos;
        if (T0 <= 0) return { freq: -1, rms };

        const freq = sampleRate / T0;
        // Human voice fundamental frequency range (roughly)
        if (freq < 60 || freq > 500) return { freq: -1, rms };
        return { freq, rms };
    }

    function drawPitchCurve() {
        if (!canvasCtx) return;
        const w = canvas.width;
        const h = canvas.height;
        canvasCtx.clearRect(0, 0, w, h);

        canvasCtx.strokeStyle = "#e5e5e5";
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, h - 1);
        canvasCtx.lineTo(w, h - 1);
        canvasCtx.stroke();

        if (pitchDrawHistory.length < 2) return;

        const maxPoints = 150;
        const recent = pitchDrawHistory.slice(-maxPoints);
        const minF = 60, maxF = 400;

        canvasCtx.strokeStyle = "#6875e8";
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();

        recent.forEach((f, i) => {
            const x = (i / (maxPoints - 1)) * w;
            const clamped = Math.max(minF, Math.min(maxF, f));
            const y = h - ((clamped - minF) / (maxF - minF)) * h;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
        });

        canvasCtx.stroke();
    }

    function mean(arr) {
        if (arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    function stdDev(arr) {
        if (arr.length < 2) return 0;
        const m = mean(arr);
        const variance = mean(arr.map((v) => (v - m) * (v - m)));
        return Math.sqrt(variance);
    }

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ---------------------------------------------------------
    // Live metric updates (called every second by the timer)
    // ---------------------------------------------------------

    function computeScores() {
        const elapsedMin = Math.max((Date.now() - sessionStartTime) / 60000, 1 / 60);
        const wpm = totalWords / elapsedMin;
        const pStd = pitchHistory.length > 5 ? stdDev(pitchHistory) : 40;

        const rateScore = recognitionSupported
            ? clamp(100 - Math.abs(wpm - 140) * 1.2, 0, 100)
            : null;
        const pitchScore = clamp(100 - Math.abs(pStd - 40) * 1.5, 0, 100);
        const pauseScore = clamp(100 - pauseCount * 8, 0, 100);
        const fillerScore = recognitionSupported
            ? clamp(100 - fillerCount * 10, 0, 100)
            : null;

        const available = [pitchScore, pauseScore];
        if (rateScore !== null) available.push(rateScore);
        if (fillerScore !== null) available.push(fillerScore);

        return {
            rateScore,
            pitchScore,
            pauseScore,
            fillerScore,
            overall: mean(available),
        };
    }

    function updateLiveMetrics() {
        const elapsedMin = Math.max((Date.now() - sessionStartTime) / 60000, 1 / 60);
        const wpm = Math.round(totalWords / elapsedMin);

        if (recognitionSupported) {
            speechRateValue.textContent = totalWords > 0 ? wpm : "--";
            if (totalWords > 0) {
                if (wpm >= 110 && wpm <= 160) {
                    speechRateStatus.textContent = "Good";
                    setStatusClass(speechRateStatus, "good");
                } else if (wpm < 110) {
                    speechRateStatus.textContent = "Too slow";
                    setStatusClass(speechRateStatus, "average");
                } else {
                    speechRateStatus.textContent = "Too fast";
                    setStatusClass(speechRateStatus, "average");
                }
            }
            fillerValue.textContent = fillerCount;
            fillerStatus.textContent = fillerCount <= 2 ? "Good" : "Frequent";
            setStatusClass(fillerStatus, fillerCount <= 2 ? "good" : "poor");
        } else {
            speechRateValue.textContent = "N/A";
            speechRateStatus.textContent = "Unsupported browser";
            fillerValue.textContent = "N/A";
            fillerStatus.textContent = "Unsupported browser";
        }

        pauseValue.textContent = pauseCount;
        pauseStatus.textContent = pauseCount <= 4 ? "Good" : "Frequent";
        setStatusClass(pauseStatus, pauseCount <= 4 ? "good" : "average");

        if (pitchHistory.length > 5) {
            const pStd = stdDev(pitchHistory.slice(-100));
            pitchValue.textContent = Math.round(mean(pitchHistory.slice(-20)));
            if (pStd >= 15 && pStd <= 70) {
                pitchStatus.textContent = "Natural tone";
                setStatusClass(pitchStatus, "good");
            } else if (pStd < 15) {
                pitchStatus.textContent = "Monotone";
                setStatusClass(pitchStatus, "average");
            } else {
                pitchStatus.textContent = "Very variable";
                setStatusClass(pitchStatus, "average");
            }
        }

        // Heuristic quality score - a placeholder until the trained Conformer
        // model is wired in via the backend (see main.py /api/analyze).
        const scores = computeScores();
        if (scores.overall >= 80) {
            qualityStatus.textContent = "GOOD";
            setStatusClass(qualityStatus, "good");
        } else if (scores.overall >= 55) {
            qualityStatus.textContent = "AVERAGE";
            setStatusClass(qualityStatus, "average");
        } else {
            qualityStatus.textContent = "NEEDS WORK";
            setStatusClass(qualityStatus, "poor");
        }
    }

    // ---------------------------------------------------------
    // Audio analysis loop
    // ---------------------------------------------------------

    function audioLoop() {
        const buffer = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(buffer);

        const { freq, rms } = autoCorrelate(buffer, audioContext.sampleRate);

        if (freq > 0) {
            pitchHistory.push(freq);
            pitchDrawHistory.push(freq);
        }

        // Pause detection via a simple silence state machine
        const now = Date.now();
        if (rms < SILENCE_RMS_THRESHOLD) {
            if (!isSilent) {
                isSilent = { counted: false, start: now };
            } else if (!isSilent.counted && now - isSilent.start > SILENCE_MIN_DURATION_MS) {
                isSilent.counted = true;
                pauseCount++;
            }
        } else {
            isSilent = false;
        }

        drawPitchCurve();
        animationFrameId = requestAnimationFrame(audioLoop);
    }

    // ---------------------------------------------------------
    // Speech recognition (speech rate + filler words)
    // ---------------------------------------------------------

    function setupSpeechRecognition() {
        const SpeechRecognitionAPI =
            window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognitionAPI) {
            recognitionSupported = false;
            showBanner(
                "info",
                "Your browser doesn't support live speech-to-text (try Chrome or Edge). " +
                "Pitch and pause detection will still work; speech rate and filler words won't."
            );
            return;
        }

        recognitionSupported = true;
        recognition = new SpeechRecognitionAPI();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onresult = function (event) {
            let interimText = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    const words = transcript.trim().split(/\s+/).filter(Boolean);
                    totalWords += words.length;
                    words.forEach((w) => {
                        const clean = w.toLowerCase().replace(/[.,!?]/g, "");
                        if (FILLER_WORDS.has(clean)) fillerCount++;
                    });
                } else {
                    interimText += transcript;
                }
            }
            if (liveTranscript) {
                liveTranscript.textContent = interimText || liveTranscript.textContent;
            }
        };

        recognition.onerror = function (event) {
            if (event.error === "not-allowed" || event.error === "service-not-allowed") {
                showBanner("error", "Microphone permission denied for speech-to-text.");
            }
            // 'no-speech' fires often and harmlessly - ignore it
        };

        // Some browsers stop recognition after silence; restart while session is active
        recognition.onend = function () {
            if (audioContext && audioContext.state !== "closed") {
                try { recognition.start(); } catch (e) { /* already running */ }
            }
        };

        try {
            recognition.start();
        } catch (e) {
            recognitionSupported = false;
        }
    }

    // ---------------------------------------------------------
    // Start / Stop handlers
    // ---------------------------------------------------------

    startAnalysis.addEventListener("click", async function () {
        // Reset state
        seconds = 0;
        pauseCount = 0;
        isSilent = false;
        pitchHistory = [];
        pitchDrawHistory = [];
        totalWords = 0;
        fillerCount = 0;
        sessionStartTime = Date.now();

        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                showBanner(
                    "error",
                    "Microphone access was denied. Please allow microphone permission and click Start again."
                );
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                showBanner("error", "No microphone was detected on this device.");
            } else {
                showBanner("error", "Could not access the microphone: " + err.message);
            }
            return;
        }

        hideBanner();

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        // Record the raw mic audio in parallel with the live analyser above,
        // so we have an actual clip to send the trained model when the user
        // stops. If MediaRecorder isn't available, we just skip this and
        // fall back to the in-browser heuristic score only.
        recordedChunks = [];
        try {
            mediaRecorder = new MediaRecorder(micStream);
            mediaRecorder.ondataavailable = function (e) {
                if (e.data && e.data.size > 0) recordedChunks.push(e.data);
            };
            mediaRecorder.start();
        } catch (e) {
            mediaRecorder = null;
        }

        setupSpeechRecognition();
        audioLoop();

        timerInterval = setInterval(function () {
            seconds++;
            timerEl.textContent = formatTimer(seconds);
            updateLiveMetrics();
        }, 1000);
    });

    stopAnalysis.addEventListener("click", async function () {
        clearInterval(timerInterval);
        cancelAnimationFrame(animationFrameId);

        if (recognition) {
            recognition.onend = null; // prevent auto-restart
            recognition.stop();
        }

        // Stop the recorder BEFORE tearing down the mic stream, and wait for
        // its final chunk, so we have a complete clip to send to the model.
        let audioBlob = null;
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            audioBlob = await new Promise(function (resolve) {
                mediaRecorder.onstop = function () {
                    resolve(new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" }));
                };
                mediaRecorder.stop();
            });
        }

        if (micStream) {
            micStream.getTracks().forEach((track) => track.stop());
        }
        if (audioContext) {
            audioContext.close();
        }

        const scores = computeScores();
        const elapsedMin = Math.max(seconds / 60, 1 / 60);
        const wpm = Math.round(totalWords / elapsedMin);
        const pitchMean = pitchHistory.length ? Math.round(mean(pitchHistory)) : 0;
        const pitchStd = pitchHistory.length ? Math.round(stdDev(pitchHistory)) : 0;

        let qualityLabel = "average";
        if (scores.overall >= 80) qualityLabel = "good";
        else if (scores.overall < 55) qualityLabel = "poor";

        const results = {
            wpm,
            pitchMean,
            pitchStd,
            pauseCount,
            fillerCount,
            durationSec: seconds,
            overallScore: Math.round(scores.overall),
            qualityLabel,
            recognitionSupported,
            rateScore: scores.rateScore !== null ? Math.round(scores.rateScore) : null,
            pitchScore: Math.round(scores.pitchScore),
            pauseScore: Math.round(scores.pauseScore),
            fillerScore: scores.fillerScore !== null ? Math.round(scores.fillerScore) : null,
            modelUsed: false,
            modelConfidence: null,
        };

        // Send the recorded clip to the real trained Conformer model. If the
        // backend is unreachable, has no trained model yet, or the request
        // fails for any reason, we silently keep the JS heuristic result
        // above rather than blocking the user from seeing their results.
        if (audioBlob && audioBlob.size > 0) {
            try {
                const formData = new FormData();
                formData.append("audio", audioBlob, "session.webm");
                const resp = await fetch("/api/analyze", { method: "POST", body: formData });
                const data = await resp.json();
                if (resp.ok && data.status === "ok") {
                    results.qualityLabel = data.quality_label;
                    results.modelUsed = true;
                    results.modelConfidence = Math.round(data.confidence * 100);
                    // Rescale the displayed score into the band matching the
                    // model's actual verdict, so overallScore and
                    // qualityLabel never contradict each other on-screen.
                    const band = { good: [80, 100], average: [55, 79], poor: [0, 54] }[data.quality_label];
                    if (band) {
                        results.overallScore = Math.round(band[0] + data.confidence * (band[1] - band[0]));
                    }
                }
            } catch (err) {
                console.warn("Model analysis unavailable, using in-browser estimate instead:", err);
            }
        }

        sessionStorage.setItem("presentationResults", JSON.stringify(results));
        window.location.href = "results.html";
    });
}


// =============================================================
// RESULTS PAGE
// =============================================================

const overallScoreEl = document.getElementById("overallScore");

if (overallScoreEl) {

    const stored = sessionStorage.getItem("presentationResults");
    const demoNotice = document.getElementById("demoNotice");

    if (stored) {
        const r = JSON.parse(stored);
        demoNotice.classList.remove("show");

        document.getElementById("overallScore").textContent = r.overallScore;

        const overallLabelEl = document.getElementById("overallLabel");
        overallLabelEl.textContent = r.qualityLabel.toUpperCase();
        overallLabelEl.classList.remove("good", "average", "poor");
        overallLabelEl.classList.add(r.qualityLabel);

        const scoreCircleEl = document.querySelector(".score-circle");
        if (scoreCircleEl) {
            scoreCircleEl.classList.remove("good", "average", "poor");
            scoreCircleEl.classList.add(r.qualityLabel);
        }

        const filledStars = Math.round(r.overallScore / 20);
        document.getElementById("starsRating").textContent =
            "★ ".repeat(filledStars) + "☆ ".repeat(5 - filledStars);

        if (r.recognitionSupported) {
            document.getElementById("rateValueText").textContent = r.wpm + " WPM";
            document.getElementById("rateStatusText").textContent =
                r.wpm >= 110 && r.wpm <= 160 ? "Good pace" : r.wpm < 110 ? "A bit slow" : "A bit fast";
            document.getElementById("rateScore").textContent = r.rateScore + "%";

            document.getElementById("fillerValueText").textContent = r.fillerCount + " Words";
            document.getElementById("fillerStatusText").textContent =
                r.fillerCount <= 2 ? "Well controlled" : "Could improve";
            document.getElementById("fillerScore").textContent = r.fillerScore + "%";
        } else {
            document.getElementById("rateValueText").textContent = "N/A";
            document.getElementById("rateStatusText").textContent = "Unsupported browser";
            document.getElementById("rateScore").textContent = "--";

            document.getElementById("fillerValueText").textContent = "N/A";
            document.getElementById("fillerStatusText").textContent = "Unsupported browser";
            document.getElementById("fillerScore").textContent = "--";
        }

        document.getElementById("pitchValueText").textContent =
            r.pitchStd >= 15 && r.pitchStd <= 70 ? "Good Variation" : r.pitchStd < 15 ? "Low Variation" : "High Variation";
        document.getElementById("pitchStatusText").textContent = r.pitchMean + " Hz average";
        document.getElementById("pitchScore").textContent = r.pitchScore + "%";

        document.getElementById("pauseValueText").textContent = r.pauseCount + " Pauses";
        document.getElementById("pauseStatusText").textContent =
            r.pauseCount <= 4 ? "Acceptable" : "Frequent";
        document.getElementById("pauseScore").textContent = r.pauseScore + "%";
    } else {
        demoNotice.classList.add("show");
    }
}


const viewRecommendations = document.getElementById("viewRecommendations");
const practiceAgain = document.getElementById("practiceAgain");

if (viewRecommendations) {
    viewRecommendations.addEventListener("click", function () {
        window.location.href = "recommendations.html";
    });
}

if (practiceAgain) {
    practiceAgain.addEventListener("click", function () {
        window.location.href = "analysis.html";
    });
}


// =============================================================
// RECOMMENDATIONS PAGE
// =============================================================

const strengthsList = document.getElementById("strengthsList");

if (strengthsList) {

    const stored = sessionStorage.getItem("presentationResults");
    const demoNotice = document.getElementById("demoNotice");
    const improveList = document.getElementById("improveList");
    const overallFeedbackText = document.getElementById("overallFeedbackText");

    if (stored) {
        const r = JSON.parse(stored);
        demoNotice.classList.remove("show");

        const metricInfo = [
            {
                score: r.rateScore, available: r.recognitionSupported,
                strengthTitle: "Good Speaking Pace",
                strengthText: "Your speech rate (" + r.wpm + " WPM) was within a comfortable range and easy to follow.",
                improveTitle: "Adjust Your Speaking Pace",
                improveText: r.wpm < 110
                    ? "Your pace (" + r.wpm + " WPM) was a little slow - try to speak with a bit more energy and momentum."
                    : "Your pace (" + r.wpm + " WPM) was a little fast - slow down slightly so your audience can follow along.",
            },
            {
                score: r.pitchScore, available: true,
                strengthTitle: "Good Pitch Variation",
                strengthText: "You maintained natural variation in your voice throughout the presentation.",
                improveTitle: r.pitchStd < 15 ? "Add More Vocal Variety" : "Watch Pitch Fluctuation",
                improveText: r.pitchStd < 15
                    ? "Your tone stayed fairly flat - try varying your pitch to emphasize key points."
                    : "Your pitch varied quite a lot - aim for controlled, natural variation rather than large swings.",
            },
            {
                score: r.pauseScore, available: true,
                strengthTitle: "Good Pause Control",
                strengthText: "Your pauses (" + r.pauseCount + ") were well placed and didn't disrupt your flow.",
                improveTitle: "Improve Pause Control",
                improveText: "You had " + r.pauseCount + " noticeable pauses. Practice connecting your ideas more smoothly to reduce hesitation.",
            },
            {
                score: r.fillerScore, available: r.recognitionSupported,
                strengthTitle: "Minimal Filler Words",
                strengthText: "You used very few filler words (" + r.fillerCount + "), which made you sound confident.",
                improveTitle: "Reduce Filler Words",
                improveText: "You used filler words like \"um\" or \"uh\" " + r.fillerCount + " times. Try pausing briefly instead of filling gaps with these words.",
            },
        ];

        let strengthsHtml = "";
        let improveHtml = "";

        metricInfo.forEach(function (m) {
            if (!m.available) return;
            if (m.score >= 75) {
                strengthsHtml += "<div class=\"recommendation-item\"><h3>" + m.strengthTitle + "</h3><p>" + m.strengthText + "</p></div>";
            } else {
                improveHtml += "<div class=\"recommendation-item\"><h3>" + m.improveTitle + "</h3><p>" + m.improveText + "</p></div>";
            }
        });

        strengthsList.innerHTML = strengthsHtml || "<div class=\"recommendation-item\"><p>Keep practicing - your strengths will show up here as you improve.</p></div>";
        improveList.innerHTML = improveHtml || "<div class=\"recommendation-item\"><p>Great job - no major areas of concern this time.</p></div>";

        overallFeedbackText.textContent =
            "You scored " + r.overallScore + "/100 (" + r.qualityLabel.toUpperCase() + "). " +
            (improveHtml
                ? "Focus on the areas below to sound even more confident next time."
                : "Your delivery was strong across the board - keep up the good habits.");

        if (!r.recognitionSupported) {
            overallFeedbackText.textContent +=
                " Note: speech rate and filler-word detection weren't available in your browser for this session.";
        }
    } else {
        demoNotice.classList.add("show");
    }
}


const practiceAgainRecommendation = document.getElementById("practiceAgainRecommendation");

if (practiceAgainRecommendation) {
    practiceAgainRecommendation.addEventListener("click", function () {
        window.location.href = "analysis.html";
    });
}


const downloadReport = document.getElementById("downloadReport");

if (downloadReport) {
    downloadReport.addEventListener("click", function () {
        // Opens the browser print dialog, styled via @media print in style.css,
        // so the user can "Save as PDF" - matches UC5 (Download Analysis Report).
        window.print();
    });
}
