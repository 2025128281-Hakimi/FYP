# How to run the Presentation Coaching System

Run every command in the VS Code terminal (Ctrl + `), using PowerShell.

---

## Run the system (every time)

**1. Start the server**

```powershell
cd "c:\Users\HP\Documents\UITM\SUBJECTS - DEGREE\FYP\source"
..\venv\Scripts\python.exe main.py
```

**2. Wait** until the terminal shows:

```
Running on http://127.0.0.1:5000
```

The first start takes about 30-60 seconds while the models load.

**3. Open the website** in Chrome: **http://127.0.0.1:5000**

**4. Stop the server** when you're done: click the terminal and press **Ctrl + C**.

Keep the terminal open while you use the website. Closing it stops the system.

---

## Using the website

1. **Analysis**: click **Start**, allow the microphone, and present.
   - The metric cards update every second: speech rate, pitch, pauses and filler words.
   - The **Live Tips** box gives one suggestion at a time, based on your last 30 seconds.
2. Click **Stop**. The Conformer model and the filler detector analyse the full recording. This takes a few seconds.
3. **Results**: your overall score, the good / average / poor rating and each metric.
4. **Recommendations**: personalised feedback written by the OpenAI API.
   Check for the line *"Personalised by AI from your measurements."*

Use **Chrome or Edge**. Firefox has no live speech-to-text, so the live speech rate and filler counts won't work there.

---

## One-time setup (new laptop, or if something is missing)

**1. Install the packages**

```powershell
cd "c:\Users\HP\Documents\UITM\SUBJECTS - DEGREE\FYP"
.\venv\Scripts\python.exe -m pip install -r source\requirements.txt
```

**2. Set the OpenAI API key** (needed for the Recommendations page)

```powershell
setx OPENAI_API_KEY "paste-your-key-here"
```

Then **close VS Code completely and open it again**. The key only works in new terminals.

Never put the key in code, in a screenshot or on GitHub. If it's ever exposed, delete it on the
OpenAI website and create a new one.

**3. Check the key is set.** This shows only the first 7 characters:

```powershell
$env:OPENAI_API_KEY.Substring(0,7)
```

If it prints something like `sk-proj`, the key is set.

---

## Files the system needs

These must exist in the `source` folder. They aren't on GitHub because they're too big:

| File | What it is |
|---|---|
| `presentation_conformer.keras` | Conformer model that rates good / average / poor |
| `filler_detector.keras` | CNN that detects "uh" / "um" |
| `filler_detector_threshold.json` | Decision settings for the filler detector |

If `presentation_conformer.keras` is missing, copy it from `Result\lr0.0001\conformer.keras` and rename it.

---

## Problems and fixes

| Problem | Fix |
|---|---|
| `No module named ...` | Run the "Install the packages" step |
| Website doesn't open | Check the terminal still shows `Running on http://127.0.0.1:5000` |
| Microphone not working | Click the lock icon in Chrome's address bar and allow the microphone |
| Recommendations: `OPENAI_API_KEY is not set` | Do the `setx` step, then restart VS Code |
| Recommendations: `the openai package is not installed` | Run the "Install the packages" step |
| Recommendations: `AI recommendations failed.` | Look at the terminal for the reason: usually no credit, no internet or an invalid key. Fix it, then click **Retry** |
| Results always say "average" with 0% confidence | The model file is missing (see "Files the system needs") |

---

## Before a demo

1. Laptop charged, internet on.
2. Start the server and open the website.
3. Do one short test recording and check the Recommendations page shows the "Personalised by AI" line.
