"use strict";

const API_URL = "http://127.0.0.1:8000";
const MAX_SEGMENTS = 3;
const POLL_INTERVAL_MS = 2000;

const $ = id => document.getElementById(id);

const themeToggle = $("theme-toggle");
const themeIcon = $("theme-icon");
const themeLabel = $("theme-label");

const vodForm = $("vod-form");
const vodUrlInput = $("vod-url");
const transcriptFileInput = $("transcript-file");

const statusMessage = $("status-message");
const resultsSection = $("results-section");
const transcriptList = $("transcript-list");
const searchInput = $("search-transcript");
const emptySearch = $("empty-search");
const transcriptCount = $("transcript-count");
const openVodLink = $("open-vod");

const submitButton = vodForm.querySelector(
    'button[type="submit"]'
);

let currentTranscript = null;
let currentVideoId = null;
let isProcessing = false;

function getCurrentTheme() {
    return document.documentElement.dataset.theme === "light"
        ? "light"
        : "dark";
}

function updateThemeButton() {
    const nextTheme =
        getCurrentTheme() === "dark"
            ? "light"
            : "dark";

    themeIcon.textContent =
        nextTheme === "light" ? "☀" : "☾";

    themeLabel.textContent =
        nextTheme === "light"
            ? "Light mode"
            : "Dark mode";

    themeToggle.title =
        `Switch to ${nextTheme} mode`;

    themeToggle.setAttribute(
        "aria-label",
        `Switch to ${nextTheme} mode`
    );

    themeToggle.setAttribute(
        "aria-pressed",
        String(getCurrentTheme() === "dark")
    );
}

themeToggle.addEventListener("click", () => {
    const nextTheme =
        getCurrentTheme() === "dark"
            ? "light"
            : "dark";

    document.documentElement.dataset.theme =
        nextTheme;

    try {
        localStorage.setItem(
            "clivra-theme",
            nextTheme
        );
    } catch {
        // Keep the theme working without storage.
    }

    updateThemeButton();
});

updateThemeButton();

function formatTime(value) {
    const total = Math.max(
        0,
        Math.floor(Number(value) || 0)
    );

    const hours = Math.floor(total / 3600);
    const minutes = Math.floor(
        (total % 3600) / 60
    );
    const seconds = total % 60;

    return [hours, minutes, seconds]
        .map(value =>
            String(value).padStart(2, "0")
        )
        .join(":");
}

function parseTime(value) {
    const match = value.trim().match(
        /^(\d{1,3}):([0-5]\d):([0-5]\d)$/
    );

    if (!match) {
        throw new Error(
            "Start time must use HH:MM:SS, " +
            "for example 00:40:10."
        );
    }

    return (
        Number(match[1]) * 3600 +
        Number(match[2]) * 60 +
        Number(match[3])
    );
}

function getTwitchVideoId(url) {
    try {
        const parsed = new URL(url);

        const hosts = [
            "twitch.tv",
            "www.twitch.tv",
            "m.twitch.tv"
        ];

        if (
            parsed.protocol !== "https:" ||
            !hosts.includes(parsed.hostname) ||
            parsed.username ||
            parsed.password ||
            parsed.port
        ) {
            return null;
        }

        const match = parsed.pathname.match(
            /^\/videos\/([0-9]+)\/?$/
        );

        return match ? match[1] : null;
    } catch {
        return null;
    }
}

function createTwitchLink(videoId, seconds) {
    const total = Math.max(
        0,
        Math.floor(Number(seconds) || 0)
    );

    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;

    return (
        `https://www.twitch.tv/videos/${videoId}` +
        `?t=${h}h${m}m${s}s`
    );
}

const timeSettings = document.createElement("div");
timeSettings.className = "analysis-settings";

const startTimeLabel = document.createElement("label");
startTimeLabel.htmlFor = "start-time";
startTimeLabel.textContent = "Start analysis at (HH:MM:SS)";

const startTimeInput = document.createElement("input");
startTimeInput.id = "start-time";
startTimeInput.type = "text";
startTimeInput.value = "00:40:10";
startTimeInput.placeholder = "00:40:10";
startTimeInput.inputMode = "numeric";

const settingsHint = document.createElement("p");
settingsHint.textContent =
    "Test mode: 3 segments × 60 seconds " +
    "(3 minutes total).";

timeSettings.append(
    startTimeLabel,
    startTimeInput,
    settingsHint
);

vodForm.insertBefore(
    timeSettings,
    submitButton
);

submitButton.replaceChildren();

const buttonText = document.createElement("span");
buttonText.textContent = "Analyze VOD";

const buttonArrow = document.createElement("span");
buttonArrow.textContent = "→";

submitButton.append(
    buttonText,
    buttonArrow
);

const progressPanel = document.createElement("div");
progressPanel.className = "analysis-progress";
progressPanel.hidden = true;

const progressHeader = document.createElement("div");
progressHeader.className = "analysis-progress-header";

const progressTitle = document.createElement("strong");
progressTitle.textContent = "Processing VOD";

const progressPercentage = document.createElement("span");
progressPercentage.textContent = "0%";

progressHeader.append(
    progressTitle,
    progressPercentage
);

const progressTrack = document.createElement("div");
progressTrack.className = "analysis-progress-track";

const progressFill = document.createElement("div");
progressFill.className = "analysis-progress-fill";
progressFill.style.width = "0%";

progressTrack.appendChild(progressFill);

const progressDetails = document.createElement("p");
progressDetails.textContent = "Preparing analysis...";

progressPanel.append(
    progressHeader,
    progressTrack,
    progressDetails
);

vodForm.after(progressPanel);


function setStatus(message, type = "normal") {
    statusMessage.textContent = message;

    statusMessage.classList.remove(
        "success",
        "error"
    );

    if (type !== "normal") {
        statusMessage.classList.add(type);
    }
}

function updateProgress(
    percentage,
    title,
    description
) {
    const value = Math.max(
        0,
        Math.min(100, percentage)
    );

    progressPanel.hidden = false;
    progressTitle.textContent = title;

    progressPercentage.textContent =
        `${Math.round(value)}%`;

    progressFill.style.width =
        `${value}%`;

    progressDetails.textContent =
        description;
}

function setProcessing(value) {
    isProcessing = value;
    submitButton.disabled = value;

    buttonText.textContent =
        value ? "Processing..." : "Analyze VOD";
}

async function apiRequest(path, options = {}) {
    let response;

    try {
        response = await fetch(
            `${API_URL}${path}`,
            options
        );
    } catch {
        throw new Error(
            "Cannot connect to FastAPI. " +
            "Make sure the backend is running " +
            "on port 8000."
        );
    }

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(
            "Backend returned an invalid response."
        );
    }

    if (!response.ok) {
        const detail = data.detail;

        throw new Error(
            typeof detail === "string"
                ? detail
                : `Request failed (${response.status})`
        );
    }

    return data;
}

function sleep(milliseconds) {
    return new Promise(resolve => {
        setTimeout(resolve, milliseconds);
    });
}

async function startAnalysis(url, startSeconds) {
    return apiRequest(
        "/api/long-vod/start",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                url,
                start_seconds: startSeconds
            })
        }
    );
}

async function getAnalysisStatus(jobId) {
    return apiRequest(
        `/api/long-vod/status/${encodeURIComponent(jobId)}`
    );
}

async function getAnalysisResult(jobId) {
    return apiRequest(
        `/api/long-vod/result/${encodeURIComponent(jobId)}`
    );
}

function showBackendProgress(status) {
    const total = MAX_SEGMENTS;

    const downloaded = Math.min(
        total,
        Number(status.download_completed) || 0
    );

    const transcribed = Math.min(
        total,
        Number(status.transcription_completed) || 0
    );

    // 50% for downloading, 50% for transcription.
    // This is progress by completed segments,
    // not a prediction of elapsed time.

    const percentage =
        (downloaded / total) * 50 +
        (transcribed / total) * 50;

    if (status.stage === "queued") {
        updateProgress(
            0,
            "Waiting",
            "Job has been queued..."
        );

    } else if (status.stage === "downloading") {
        updateProgress(
            percentage,
            "Downloading VOD",
            `${downloaded}/${total} video segments completed`
        );

    } else if (status.stage === "transcribing") {
        updateProgress(
            percentage,
            "Transcribing Audio",
            `${transcribed}/${total} audio segments transcribed`
        );

    } else if (status.status === "completed") {
        updateProgress(
            100,
            "Analysis Complete",
            "Transcript is ready."
        );
    }
}

async function waitForAnalysis(jobId) {
    while (true) {
        const status = await getAnalysisStatus(
            jobId
        );

        showBackendProgress(status);

        if (status.status === "completed") {
            return getAnalysisResult(jobId);
        }

        if (
            status.status === "failed" ||
            status.status === "interrupted"
        ) {
            throw new Error(
                status.error ||
                status.message ||
                "VOD processing failed."
            );
        }

        await sleep(POLL_INTERVAL_MS);
    }
}

function validateTranscript(data) {
    if (
        !data ||
        typeof data !== "object" ||
        !/^[0-9]+$/.test(
            String(data.video_id ?? "")
        ) ||
        !Array.isArray(data.segments)
    ) {
        throw new Error(
            "Invalid Clivra transcript format."
        );
    }

    const segments = data.segments.filter(
        item =>
            item &&
            typeof item.text === "string" &&
            Number.isFinite(
                Number(item.vod_start)
            ) &&
            Number(item.vod_start) >= 0
    );

    segments.sort(
        (a, b) =>
            Number(a.vod_start) -
            Number(b.vod_start)
    );

    return {
        ...data,
        video_id: String(data.video_id),
        segments
    };
}

function createTranscriptItem(segment) {
    const item = document.createElement("div");
    item.className = "transcript-item";

    const timestamp = document.createElement("a");
    timestamp.className = "timestamp-link";

    timestamp.textContent = formatTime(
        segment.vod_start
    );

    timestamp.href = createTwitchLink(
        currentVideoId,
        segment.vod_start
    );

    timestamp.target = "_blank";
    timestamp.rel = "noopener noreferrer";
    timestamp.title = "Open timestamp on Twitch";

    const text = document.createElement("p");
    text.className = "transcript-text";
    text.textContent = segment.text;

    item.append(timestamp, text);

    return item;
}

function renderTranscript() {
    if (!currentTranscript) {
        return;
    }

    const query = searchInput.value
        .trim()
        .toLowerCase();

    const matching =
        currentTranscript.segments.filter(
            segment =>
                segment.text
                    .toLowerCase()
                    .includes(query) ||
                formatTime(segment.vod_start)
                    .includes(query)
        );

    const fragment =
        document.createDocumentFragment();

    for (const segment of matching) {
        fragment.appendChild(
            createTranscriptItem(segment)
        );
    }

    transcriptList.replaceChildren(fragment);

    transcriptCount.textContent =
        `${matching.length} results`;

    emptySearch.hidden =
        matching.length !== 0;
}

function showResults(rawData) {
    const data = validateTranscript(
        rawData
    );

    currentTranscript = data;
    currentVideoId = data.video_id;

    const title =
        typeof data.title === "string" &&
        data.title.trim()
            ? data.title
            : `Twitch VOD #${data.video_id}`;

    $("video-title").textContent = title;

    const streamer =
        typeof data.streamer === "string" &&
        data.streamer.trim()
            ? data.streamer
            : "Unknown streamer";

    $("video-subtitle").textContent =
        `${streamer} · Video ID: ${data.video_id}`;

    $("stat-duration").textContent =
        formatTime(data.vod_duration);

    $("stat-words").textContent =
        Number(data.total_words ?? 0)
            .toLocaleString();

    $("stat-segments").textContent =
        Number(
            data.total_speech_segments ??
            data.segments.length
        ).toLocaleString();

    $("stat-processed").textContent =
        `${data.transcribed_vod_segments ?? 0}/` +
        `${data.total_vod_segments ?? 0}`;

    openVodLink.href =
        `https://www.twitch.tv/videos/${data.video_id}`;

    vodUrlInput.value =
        `https://www.twitch.tv/videos/${data.video_id}`;

    searchInput.value = "";
    resultsSection.hidden = false;

    renderTranscript();

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

vodForm.addEventListener(
    "submit",
    async event => {
        event.preventDefault();

        if (isProcessing) {
            return;
        }

        const url = vodUrlInput.value.trim();
        const videoId = getTwitchVideoId(url);

        if (!videoId) {
            setStatus(
                "Please enter a valid Twitch VOD URL.",
                "error"
            );
            return;
        }

        let startSeconds;

        try {
            startSeconds = parseTime(
                startTimeInput.value
            );
        } catch (error) {
            setStatus(
                error.message,
                "error"
            );
            return;
        }

        setProcessing(true);
        resultsSection.hidden = true;

        updateProgress(
            0,
            "Starting Analysis",
            "Connecting to Clivra backend..."
        );

        setStatus(
            "Clivra is processing your Twitch VOD."
        );

        try {
            const job = await startAnalysis(
                url,
                startSeconds
            );

            const transcript =
                await waitForAnalysis(
                    job.job_id
                );

            showResults(transcript);

            setStatus(
                "VOD transcription completed successfully.",
                "success"
            );

        } catch (error) {
            progressPanel.hidden = true;

            setStatus(
                error.message ||
                "Failed to analyze VOD.",
                "error"
            );

        } finally {
            setProcessing(false);
        }
    }
);

transcriptFileInput.addEventListener(
    "change",
    async event => {
        const file = event.target.files?.[0];

        if (!file) {
            return;
        }

        try {
            if (file.size > 20 * 1024 * 1024) {
                throw new Error(
                    "JSON file exceeds the 20 MB limit."
                );
            }

            const data = validateTranscript(
                JSON.parse(await file.text())
            );

            const enteredUrl =
                vodUrlInput.value.trim();

            if (enteredUrl) {
                const enteredId =
                    getTwitchVideoId(enteredUrl);

                if (
                    !enteredId ||
                    enteredId !== data.video_id
                ) {
                    throw new Error(
                        "The URL and transcript " +
                        "must belong to the same Twitch VOD."
                    );
                }
            }

            showResults(data);
            progressPanel.hidden = true;

            setStatus(
                `Imported ${data.segments.length} ` +
                "speech segments.",
                "success"
            );

        } catch (error) {
            setStatus(
                error.message,
                "error"
            );

        } finally {
            event.target.value = "";
        }
    }
);

searchInput.addEventListener(
    "input",
    renderTranscript
);
