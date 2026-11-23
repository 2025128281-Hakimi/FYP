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
    const tipMain = document.getElementById("tip1");
    const tipDetail = document.getElementById("tip2");
    const tipFooter = document.getElementById("tip3");
    const canvas = document.getElementById("pitchCanvas");
    const canvasCtx = canvas ? canvas.getContext("2d") : null;

    // Hesitation sounds, including stretched spellings like "ummm", "uhhh", "erm", "aaa", "hmm".
    const FILLER_SOUND = /^(u+h+|u+m+|u+h+m+|e+r+|e+r+m+|a+h+|aa+|h+m+|mm+|e+h+)$/;

    // "like" is only a filler when it isn't used as a verb or comparison ("I like", "looks like").
    const LIKE_NOT_FILLER_BEFORE = new Set([
        "i", "you", "we", "they", "he", "she", "would", "i'd", "you'd", "we'd",
        "don't", "didn't", "doesn't", "really", "look", "looks", "looked",
        "feel", "feels", "felt", "sound", "sounds", "seem", "seems", "just", "more", "much"
    ]);

    // "you know" is a question, not a filler, after these words ("do you know ...").
    const YOU_KNOW_NOT_FILLER_BEFORE = new Set([
        "do", "did", "don't", "didn't", "if", "what", "how", "would", "will", "can"
    ]);

    
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

    let pitchHistory = [];      
    let pitchDrawHistory = [];   

    let totalWords = 0;
    let fillerCount = 0;
    let pendingInterim = "";
    let pendingDraftFillers = 0;   // fillers in the sentence still being spoken
    let draftFillerMax = {};       // result index -> most fillers seen in its drafts
    let sessionStartTime = 0;

    // Timestamped events for the live tips, which only look at the last 30 seconds
    // so a tip disappears once the speaker fixes the problem.
    let wordEvents = [];     // { t, n } - n words recognised at time t
    let fillerTimes = [];
    let pauseTimes = [];
    let pitchSamples = [];   // { t, f }
    let lastSoundTime = 0;
    let currentTipKey = "";
    let lastTipChange = 0;

    
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

        const offset = maxPoints - recent.length;
        recent.forEach((f, i) => {
            const x = ((offset + i) / (maxPoints - 1)) * w;
            const clamped = f === null ? minF : Math.max(minF, Math.min(maxF, f));
            const y = Math.min(h - 2, h - ((clamped - minF) / (maxF - minF)) * h);
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
            // Include fillers in the sentence still being spoken, so the counter
            // reacts straight away instead of waiting for the sentence to end.
            const liveFillers = fillerCount + pendingDraftFillers;
            fillerValue.textContent = liveFillers;
            fillerStatus.textContent = liveFillers <= 2 ? "Good" : "Frequent";
            setStatusClass(fillerStatus, liveFillers <= 2 ? "good" : "poor");
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

        updateLiveTips();
    }

    // ---------------------------------------------------------
    // Live tips (rule-based so they react instantly; the detailed
    // feedback after the session comes from the OpenAI API)
    // ---------------------------------------------------------

    const TIP_WINDOW_MS = 30000;   // judge only the last 30 s
    const TIP_WARMUP_MS = 15000;   // no tips until there's enough speech to judge
    const TIP_HOLD_MS = 10000;     // keep each tip on screen at least this long
    const LONG_SILENCE_MS = 5000;

    function setTip(main, detail, footer) {
        tipMain.textContent = main;
        tipDetail.textContent = detail;
        tipFooter.textContent = footer;
    }

    function resetTips() {
        currentTipKey = "";
        lastTipChange = 0;
        tipMain.classList.remove("tip-good", "tip-warn");
        setTip("Listening...", "Your first tip appears after about 15 seconds of speaking.", "");
    }

    // Returns the most important problem in the last 30 s (or a "good" message).
    // Order = priority: the first rule that matches wins.
    function pickTip(now) {
        const windowStart = Math.max(sessionStartTime, now - TIP_WINDOW_MS);
        const windowMin = (now - windowStart) / 60000;
        const words = wordEvents.filter((e) => e.t >= windowStart).reduce((s, e) => s + e.n, 0);
        const wpm = Math.round(words / windowMin);
        const fillers = fillerTimes.filter((t) => t >= windowStart).length + pendingDraftFillers;
        const pauses = pauseTimes.filter((t) => t >= windowStart).length;
        const pitches = pitchSamples.filter((p) => p.t >= windowStart).map((p) => p.f);
        const pStd = pitches.length > 30 ? stdDev(pitches) : null;

        if (now - lastSoundTime > LONG_SILENCE_MS) {
            return { key: "silence", level: "warn",
                     main: "Keep going - move on to your next point",
                     detail: "You've been silent for " + Math.round((now - lastSoundTime) / 1000) + " seconds." };
        }
        if (recognitionSupported && fillers >= 3) {
            return { key: "fillers", level: "warn",
                     main: "Pause silently instead of saying \"um\" or \"uh\"",
                     detail: fillers + " filler words in the last 30 seconds." };
        }
        if (recognitionSupported && words >= 10 && wpm > 160) {
            return { key: "fast", level: "warn",
                     main: "Slow down - take a breath between sentences",
                     detail: "Current pace: " + wpm + " words per minute (aim for 110-160)." };
        }
        // words >= 5: zero words usually means speech-to-text is lagging, not slow speech.
        if (recognitionSupported && words >= 5 && wpm < 110) {
            return { key: "slow", level: "warn",
                     main: "Pick up the pace a little",
                     detail: "Current pace: " + wpm + " words per minute (aim for 110-160)." };
        }
        if (pStd !== null && pStd < 15) {
            return { key: "monotone", level: "warn",
                     main: "Raise your voice on key words",
                     detail: "Your tone has been quite flat - vary your pitch to stay engaging." };
        }
        if (pStd !== null && pStd > 70) {
            return { key: "variable", level: "warn",
                     main: "Keep your voice steady",
                     detail: "Your pitch is swinging a lot - aim for calm, natural variation." };
        }
        if (pauses >= 6) {
            return { key: "pauses", level: "warn",
                     main: "Connect your ideas more smoothly",
                     detail: pauses + " hesitation pauses in the last 30 seconds." };
        }
        return { key: "good", level: "good",
                 main: "Great delivery - keep it up!",
                 detail: recognitionSupported
                     ? "Pace " + wpm + " WPM, " + fillers + " filler words in the last 30 seconds."
                     : "Your pitch and pauses look good." };
    }

    function updateLiveTips() {
        const now = Date.now();
        const elapsed = now - sessionStartTime;

        // Drop events that are too old to matter.
        const cutoff = now - TIP_WINDOW_MS;
        wordEvents = wordEvents.filter((e) => e.t >= cutoff);
        fillerTimes = fillerTimes.filter((t) => t >= cutoff);
        pauseTimes = pauseTimes.filter((t) => t >= cutoff);
        pitchSamples = pitchSamples.filter((p) => p.t >= cutoff);

        if (elapsed < TIP_WARMUP_MS) {
            const left = Math.ceil((TIP_WARMUP_MS - elapsed) / 1000);
            setTip("Listening...", "Your first tip appears in " + left + " seconds.", "");
            return;
        }

        const tip = pickTip(now);
        // The silence tip is urgent and its timer keeps counting, so it may update
        // straight away; any other change waits until the current tip has had its time.
        const canChange = currentTipKey === "" || tip.key === "silence" ||
            currentTipKey === "silence" || now - lastTipChange >= TIP_HOLD_MS;
        if (tip.key !== currentTipKey && !canChange) return;

        if (tip.key !== currentTipKey) {
            currentTipKey = tip.key;
            lastTipChange = now;
        }
        tipMain.classList.toggle("tip-good", tip.level === "good");
        tipMain.classList.toggle("tip-warn", tip.level === "warn");
        setTip(tip.main, tip.detail, "Based on your last 30 seconds.");
    }

    // ---------------------------------------------------------
    // Audio analysis loop
    // ---------------------------------------------------------

    function audioLoop() {
        const buffer = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(buffer);

        const { freq, rms } = autoCorrelate(buffer, audioContext.sampleRate);

        const now = Date.now();
        if (freq > 0) {
            pitchHistory.push(freq);
            pitchSamples.push({ t: now, f: freq });
        }
        pitchDrawHistory.push(freq > 0 ? freq : null);
        if (pitchDrawHistory.length > 150) pitchDrawHistory.shift();

        if (rms < SILENCE_RMS_THRESHOLD) {
            if (!isSilent) {
                isSilent = { counted: false, start: now };
            } else if (!isSilent.counted && now - isSilent.start > SILENCE_MIN_DURATION_MS) {
                isSilent.counted = true;
                pauseCount++;
                pauseTimes.push(now);
            }
        } else {
            isSilent = false;
            lastSoundTime = now;
        }

        drawPitchCurve();
        animationFrameId = requestAnimationFrame(audioLoop);
    }

    // ---------------------------------------------------------
    // Speech recognition (speech rate + filler words)
    // ---------------------------------------------------------

    function splitWords(text) {
        return text
            .toLowerCase()
            .split(/\s+/)
            .map((w) => w.replace(/[^a-z']/g, ""))
            .filter(Boolean);
    }

    function countFillers(words) {
        let n = 0;
        words.forEach((w, i) => {
            const prev = i > 0 ? words[i - 1] : "";
            if (FILLER_SOUND.test(w) ||
                (w === "like" && !LIKE_NOT_FILLER_BEFORE.has(prev)) ||
                (w === "know" && prev === "you" &&
                 !YOU_KNOW_NOT_FILLER_BEFORE.has(i > 1 ? words[i - 2] : ""))) {
                n++;
            }
        });
        return n;
    }

    // Chrome shows "uh"/"um" in its draft (interim) text but usually deletes them
    // from the final text. draftFillers is the most fillers any draft of the
    // sentence contained, so those fillers still get counted.
    function countFinalText(text, draftFillers) {
        const words = splitWords(text);
        totalWords += words.length;
        const now = Date.now();
        if (words.length) wordEvents.push({ t: now, n: words.length });

        const fillers = Math.max(countFillers(words), draftFillers || 0);
        for (let k = 0; k < fillers; k++) {
            fillerCount++;
            fillerTimes.push(now);
        }
    }

    // Speech the browser heard but never marked "final" (e.g. recognition
    // restarted or Stop was clicked mid-sentence) would otherwise be lost.
    function flushPendingInterim() {
        if (pendingInterim.trim() || pendingDraftFillers > 0) {
            countFinalText(pendingInterim, pendingDraftFillers);
        }
        pendingInterim = "";
        pendingDraftFillers = 0;
        // Result indexes start again from 0 after recognition restarts.
        draftFillerMax = {};
    }

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
            let interimFillers = 0;
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    countFinalText(transcript, draftFillerMax[i]);
                    delete draftFillerMax[i];
                } else {
                    interimText += transcript;
                    draftFillerMax[i] = Math.max(draftFillerMax[i] || 0,
                                                 countFillers(splitWords(transcript)));
                    interimFillers += draftFillerMax[i];
                }
            }
            pendingInterim = interimText;
            pendingDraftFillers = interimFillers;
            if (liveTranscript) {
                liveTranscript.textContent = interimText || liveTranscript.textContent;
                // Keep the newest words in view when the sentence is longer than 3 lines.
                liveTranscript.scrollTop = liveTranscript.scrollHeight;
            }
        };

        recognition.onerror = function (event) {
            if (event.error === "not-allowed" || event.error === "service-not-allowed") {
                showBanner("error", "Microphone permission denied for speech-to-text.");
            }
          
        };

       
        recognition.onend = function () {
            flushPendingInterim();
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
        
        seconds = 0;
        pauseCount = 0;
        isSilent = false;
        pitchHistory = [];
        pitchDrawHistory = [];
        totalWords = 0;
        fillerCount = 0;
        pendingInterim = "";
        pendingDraftFillers = 0;
        draftFillerMax = {};
        sessionStartTime = Date.now();
        wordEvents = [];
        fillerTimes = [];
        pauseTimes = [];
        pitchSamples = [];
        lastSoundTime = sessionStartTime;
        resetTips();

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

        stopAnalysis.disabled = true;
        stopAnalysis.textContent = "Analysing...";
        showBanner("info", "Analysing your speech with the AI model. This can take a few seconds for longer recordings...");

        // Wait (max 1.5 s) for the browser to deliver the last sentence,
        // otherwise filler words in it are never counted.
        if (recognition) {
            await new Promise(function (resolve) {
                const timeout = setTimeout(resolve, 1500);
                recognition.onend = function () {
                    clearTimeout(timeout);
                    resolve();
                };
                recognition.stop();
            });
            flushPendingInterim();
        }

        
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

       
        if (audioBlob && audioBlob.size > 0) {
            try {
                const formData = new FormData();
                formData.append("audio", audioBlob, "session.webm");
                const resp = await fetch("/api/analyze", { method: "POST", body: formData });
                const data = await resp.json();

                // The server's filler detector listens to the audio itself, so it catches
                // the "um"/"uh" that browser speech-to-text drops. Prefer its count.
                if (resp.ok && typeof data.filler_count === "number") {
                    results.fillerCount = data.filler_count;
                    results.fillerScore = Math.round(clamp(100 - data.filler_count * 10, 0, 100));
                    results.fillerDetectedBy = "model";
                    const available = [results.rateScore, results.pitchScore, results.pauseScore, results.fillerScore]
                        .filter((s) => s !== null);
                    results.overallScore = Math.round(mean(available));
                }

                if (resp.ok && data.status === "ok") {
                    results.qualityLabel = data.quality_label;
                    results.modelUsed = true;
                    results.modelConfidence = Math.round(data.confidence * 100);
                 
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

function renderRecommendationItems(container, items, emptyText) {
    container.innerHTML = "";
    if (!items.length) {
        items = [{ title: "", text: emptyText }];
    }
    items.forEach(function (item) {
        const div = document.createElement("div");
        div.className = "recommendation-item";
        if (item.title) {
            const h3 = document.createElement("h3");
            h3.textContent = item.title;
            div.appendChild(h3);
        }
        const p = document.createElement("p");
        p.textContent = item.text;
        div.appendChild(p);
        container.appendChild(div);
    });
}

// All recommendations are written by the server's OpenAI endpoint - there is no
// hardcoded fallback. If the call fails, the page says why and offers a retry.
async function loadAiRecommendations(r, improveList, overallFeedbackText, note) {
    const scoreText = "You scored " + r.overallScore + "/100 (" + r.qualityLabel.toUpperCase() + "). ";

    let ai = r.aiRecommendations;
    if (!ai) {
        overallFeedbackText.textContent = scoreText;
        note.textContent = "Generating personalised feedback...";
        strengthsList.innerHTML = "";
        improveList.innerHTML = "";
        try {
            const resp = await fetch("/api/recommendations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(r),
            });
            const data = await resp.json().catch(function () { return {}; });
            if (!resp.ok) throw new Error(data.error || "Server error (HTTP " + resp.status + ").");
            ai = data;
            r.aiRecommendations = ai;
            sessionStorage.setItem("presentationResults", JSON.stringify(r));
        } catch (err) {
            console.error("AI recommendations failed:", err);
            note.textContent = "Couldn't generate your recommendations: " + err.message + " ";
            const retry = document.createElement("button");
            retry.type = "button";
            retry.className = "ai-retry";
            retry.textContent = "Retry";
            retry.addEventListener("click", function () {
                loadAiRecommendations(r, improveList, overallFeedbackText, note);
            });
            note.appendChild(retry);
            return;
        }
    }

    renderRecommendationItems(strengthsList, ai.strengths,
        "Keep practicing - your strengths will show up here as you improve.");
    renderRecommendationItems(improveList, ai.improvements,
        "Great job - no major areas of concern this time.");
    overallFeedbackText.textContent = scoreText + (ai.overall || "");
    note.textContent = "Personalised by AI from your measurements.";
}

if (strengthsList) {

    const stored = sessionStorage.getItem("presentationResults");
    const demoNotice = document.getElementById("demoNotice");
    const improveList = document.getElementById("improveList");
    const overallFeedbackText = document.getElementById("overallFeedbackText");

    if (stored) {
        const r = JSON.parse(stored);
        demoNotice.classList.remove("show");

        const note = document.createElement("p");
        note.className = "ai-note";
        overallFeedbackText.after(note);

        loadAiRecommendations(r, improveList, overallFeedbackText, note);
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
      
        window.print();
    });
}
