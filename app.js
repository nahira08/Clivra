
"use strict";

// Clivra - Milestone 7A
// Frontend transcript viewer

const vodForm = document.getElementById("vod-form");
const vodUrlInput = document.getElementById("vod-url");
const transcriptFileInput =
    document.getElementById("transcript-file");

const statusMessage =
    document.getElementById("status-message");

const resultsSection =
    document.getElementById("results-section");

const transcriptList =
    document.getElementById("transcript-list");

const searchInput =
    document.getElementById("search-transcript");

const emptySearch =
    document.getElementById("empty-search");

const transcriptCount =
    document.getElementById("transcript-count");

const openVodLink =
    document.getElementById("open-vod");

let currentVideoId = null;
let currentTranscript = null;


// Validate Twitch VOD URL

function getTwitchVideoId(url) {
    try {
        const parsed = new URL(url);

        const allowedHosts = [
            "twitch.tv",
            "www.twitch.tv",
            "m.twitch.tv"
        ];

        if (
            parsed.protocol !== "https:" ||
            !allowedHosts.includes(parsed.hostname) ||
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


// Format seconds into HH:MM:SS

function formatTime(totalSeconds) {
    const seconds = Math.max(
        0,
        Math.floor(Number(totalSeconds) || 0)
    );

    const hours = Math.floor(seconds / 3600);

    const minutes = Math.floor(
        (seconds % 3600) / 60
    );

    const remaining = seconds % 60;

    return [
        hours,
        minutes,
        remaining
    ]
        .map(number => String(number).padStart(2, "0"))
        .join(":");
}


// Create Twitch URL with timestamp

function createTwitchLink(videoId, seconds) {
    const safeSeconds = Math.max(
        0,
        Math.floor(Number(seconds) || 0)
    );

    const hours = Math.floor(safeSeconds / 3600);
    const minutes = Math.floor(
        (safeSeconds % 3600) / 60
    );
    const remaining = safeSeconds % 60;

    const timeParameter =
        `${hours}h${minutes}m${remaining}s`;

    return (
        `https://www.twitch.tv/videos/${videoId}` +
        `?t=${timeParameter}`
    );
}


// Display feedback

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


// Validate transcript JSON

function validateTranscript(data) {
    if (
        !data ||
        typeof data !== "object" ||
        !/^[0-9]+$/.test(String(data.video_id ?? "")) ||
        !Array.isArray(data.segments)
    ) {
        throw new Error(
            "Invalid transcript format. " +
            "Please select a Clivra merged_transcript.json file."
        );
    }

    const validSegments = data.segments.filter(
        segment =>
            segment &&
            typeof segment.text === "string" &&
            Number.isFinite(Number(segment.vod_start)) &&
            Number(segment.vod_start) >= 0
    );

    return {
        ...data,
        video_id: String(data.video_id),
        segments: validSegments.sort(
            (a, b) =>
                Number(a.vod_start) -
                Number(b.vod_start)
        )
    };
}


// Create transcript row safely

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

    timestamp.title =
        "Open this timestamp on Twitch";

    const text = document.createElement("p");
    text.className = "transcript-text";
    text.textContent = segment.text;

    item.appendChild(timestamp);
    item.appendChild(text);

    return item;
}


// Render filtered transcript

function renderTranscript() {
    if (!currentTranscript || !currentVideoId) {
        return;
    }

    const searchTerm = searchInput.value
        .trim()
        .toLowerCase();

    const filtered = currentTranscript.segments.filter(
        segment =>
            segment.text.toLowerCase().includes(
                searchTerm
            ) ||
            formatTime(segment.vod_start).includes(
                searchTerm
            )
    );

    transcriptList.replaceChildren();

    const fragment = document.createDocumentFragment();

    for (const segment of filtered) {
        fragment.appendChild(
            createTranscriptItem(segment)
        );
    }

    transcriptList.appendChild(fragment);

    transcriptCount.textContent =
        `${filtered.length} results`;

    emptySearch.hidden = filtered.length > 0;
}


// Render overview and statistics

function showResults(data) {
    currentTranscript = data;
    currentVideoId = data.video_id;

    const title =
        typeof data.title === "string" &&
        data.title.trim()
            ? data.title
            : `Twitch VOD #${data.video_id}`;

    document.getElementById(
        "video-title"
    ).textContent = title;

    const streamer =
        typeof data.streamer === "string" &&
        data.streamer.trim()
            ? data.streamer
            : "Unknown streamer";

    document.getElementById(
        "video-subtitle"
    ).textContent =
        `${streamer} · Video ID: ${data.video_id}`;

    document.getElementById(
        "stat-duration"
    ).textContent = formatTime(
        data.vod_duration
    );

    document.getElementById(
        "stat-words"
    ).textContent = Number(
        data.total_words ?? 0
    ).toLocaleString();

    document.getElementById(
        "stat-segments"
    ).textContent = Number(
        data.total_speech_segments ??
        data.segments.length
    ).toLocaleString();

    const processed = Number(
        data.transcribed_vod_segments ?? 0
    );

    const total = Number(
        data.total_vod_segments ?? 0
    );

    document.getElementById(
        "stat-processed"
    ).textContent = `${processed}/${total}`;

    openVodLink.href =
        `https://www.twitch.tv/videos/${data.video_id}`;

    searchInput.value = "";

    resultsSection.hidden = false;

    renderTranscript();

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// URL form behavior

vodForm.addEventListener("submit", event => {
    event.preventDefault();

    const videoId = getTwitchVideoId(
        vodUrlInput.value.trim()
    );

    if (!videoId) {
        setStatus(
            "Please enter a valid Twitch VOD URL.",
            "error"
        );

        return;
    }

    if (
        currentTranscript &&
        currentTranscript.video_id === videoId
    ) {
        showResults(currentTranscript);

        setStatus(
            "Transcript loaded successfully.",
            "success"
        );

        return;
    }

    resultsSection.hidden = true;

    // Milestone 7A does not start backend processing.
    setStatus(
        `VOD ${videoId} is valid. ` +
        "Now import its merged_transcript.json " +
        "to display the transcript."
    );
});


// Import merged_transcript.json

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
                    "Transcript file exceeds the 20 MB " +
                    "preview limit."
                );
            }

            const rawText = await file.text();
            const rawData = JSON.parse(rawText);

            const data = validateTranscript(
                rawData
            );

            const enteredUrl =
                vodUrlInput.value.trim();

            if (enteredUrl) {
                const enteredVideoId =
                    getTwitchVideoId(enteredUrl);

                if (!enteredVideoId) {
                    throw new Error(
                        "The Twitch URL is invalid. " +
                        "Please correct it before importing."
                    );
                }

                if (
                    enteredVideoId !==
                    data.video_id
                ) {
                    throw new Error(
                        "This transcript belongs to a " +
                        "different Twitch VOD."
                    );
                }
            }

            vodUrlInput.value =
                `https://www.twitch.tv/videos/` +
                data.video_id;

            showResults(data);

            const message =
                data.segments.length > 0
                    ? (
                        `Imported ${data.segments.length} ` +
                        "speech segments successfully."
                    )
                    : (
                        "Transcript imported, but no " +
                        "speech was detected in the " +
                        "processed segments."
                    );

            setStatus(
                message,
                "success"
            );

        } catch (error) {
            setStatus(
                error.message ||
                "Unable to read transcript file.",
                "error"
            );

        } finally {
            // Allow importing the same file again
            event.target.value = "";
        }
    }
);


// Search events

searchInput.addEventListener(
    "input",
    renderTranscript
);
