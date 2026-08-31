// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// /traffic cascade demo frontend: WebRTC video (recvonly) where detection
// boxes are already burned into the frames server-side - no canvas overlay
// needed. Auto-connects on page load. A WebSocket carries throttled
// detection updates (label, score, cropped thumbnail image) for the
// "Detected Vehicles" side panel. Phase 2 scope only - no click-to-caption
// yet (that's Phase 3).

(function () {
    const videoElement = document.getElementById('trafficVideo');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const connectionStatus = document.getElementById('connectionStatus');
    const trackedList = document.getElementById('trackedList');
    const trackedCount = document.getElementById('trackedCount');
    const toggleBtn = document.getElementById('toggleBtn');
    const fpsCounter = document.getElementById('fpsCounter');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const editPromptBtn = document.getElementById('editPromptBtn');
    const promptEditArea = document.getElementById('promptEditArea');
    const promptEditTextarea = document.getElementById('promptEditTextarea');
    const promptSaveBtn = document.getElementById('promptSaveBtn');
    const promptResetBtn = document.getElementById('promptResetBtn');
    const promptCancelBtn = document.getElementById('promptCancelBtn');

    let peerConnection = null;
    let ws = null;
    let reconnectTimer = null;
    let isReconnecting = false;
    let reconnectAttempts = 0;
    let connectionStableTimer = null;
    // True only when the Stop button caused the disconnect - distinguishes
    // an intentional stop from an unexpected drop, so auto-reconnect knows
    // not to fight the user's own Stop click.
    let userInitiatedStop = false;
    const RECONNECT_BASE_DELAY_MS = 3000;
    const RECONNECT_MAX_DELAY_MS = 30000;

    // Append-only history of confirmed vehicle sightings, newest first -
    // each vehicle is reported by the server exactly once (see tracker.py),
    // so entries are never replaced/deduped client-side, only capped for
    // display. totalDetectionCount is the true running total (never
    // trimmed) shown in the header badge; the list itself only ever shows
    // the most recent MAX_HISTORY.
    const MAX_HISTORY = 40;
    let history = [];
    let totalDetectionCount = 0;

    // Click-to-caption prompt - server-authoritative (see traffic_caption_prompt
    // in server.py), synced in on the initial "status" message and whenever
    // update_prompt is broadcast (including our own edits, echoed back).
    let currentPrompt = '';
    let defaultPrompt = '';

    function updateStatus(text, state) {
        connectionStatus.textContent = text;
        connectionStatus.className = 'status-badge ' + state;
    }

    // Actual rendered playback rate. getVideoPlaybackQuality()'s
    // totalVideoFrames was tried first but doesn't reliably increment for a
    // live WebRTC MediaStream srcObject in practice (it's really geared
    // towards <video src="file"> playback) - requestVideoFrameCallback
    // fires once per actually-rendered frame regardless of source and is
    // the reliable way to measure this for a live stream.
    let frameCount = 0;
    if (videoElement.requestVideoFrameCallback) {
        const onFrame = () => {
            frameCount++;
            videoElement.requestVideoFrameCallback(onFrame);
        };
        videoElement.requestVideoFrameCallback(onFrame);

        setInterval(() => {
            fpsCounter.textContent = peerConnection ? `${frameCount} FPS` : '-- FPS';
            frameCount = 0;
        }, 1000);
    } else {
        fpsCounter.textContent = 'FPS n/a';
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str == null ? '' : str;
        return div.innerHTML;
    }

    function renderHistory() {
        trackedCount.textContent = totalDetectionCount;

        if (history.length === 0) {
            trackedList.innerHTML = '<div class="tracked-empty">Waiting for detections...</div>';
            return;
        }

        trackedList.innerHTML = history
            .map((obj) => {
                const thumb = obj.crop
                    ? `<img class="tracked-item-thumb" src="data:image/jpeg;base64,${obj.crop}" alt="${obj.label}">`
                    : `<div class="tracked-item-thumb"></div>`;

                const tag =
                    obj.captionStatus === 'done'
                        ? `<span class="tracked-item-tag">&#10003; Described</span>`
                        : obj.captionStatus === 'pending'
                          ? `<span class="tracked-item-tag">Describing...</span>`
                          : '';

                return `
                <div class="tracked-item" data-id="${obj.id}">
                    ${thumb}
                    <div class="tracked-item-info">
                        <span class="tracked-item-label">${escapeHtml(obj.label)}</span>
                        <span class="tracked-item-score">${(obj.score * 100).toFixed(0)}% confidence</span>
                        ${tag}
                    </div>
                </div>`;
            })
            .join('');
    }

    // Event delegation on the (re-rendered) list container, so click
    // handling survives renderHistory() replacing the list's innerHTML.
    // Clicking a card opens the detail modal (see below) rather than
    // captioning inline - inline captions got visually cramped/pushed
    // around as new cards keep arriving while the video runs.
    trackedList.addEventListener('click', (event) => {
        const card = event.target.closest('.tracked-item');
        if (card) openModal(card.dataset.id);
    });

    // --- Detail modal: larger image + Describe action ---
    const detailModalBackdrop = document.getElementById('detailModalBackdrop');
    const detailModalImg = document.getElementById('detailModalImg');
    const detailModalLabel = document.getElementById('detailModalLabel');
    const detailModalScore = document.getElementById('detailModalScore');
    const detailModalCaptionArea = document.getElementById('detailModalCaptionArea');
    const detailModalClose = document.getElementById('detailModalClose');
    let openModalId = null;

    function requestCaption(id) {
        const entry = history.find((h) => h.id === id);
        if (entry) {
            entry.captionStatus = 'pending';
            renderHistory();
            if (openModalId === id) renderModalContent();
        }
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'request_caption', id }));
        }
    }

    function renderModalContent() {
        const entry = history.find((h) => h.id === openModalId);
        if (!entry) return;

        const modalImg = entry.crop_large || entry.crop;
        detailModalImg.src = modalImg ? `data:image/jpeg;base64,${modalImg}` : '';
        detailModalImg.alt = entry.label;
        detailModalLabel.textContent = entry.label;
        detailModalScore.textContent = `${(entry.score * 100).toFixed(0)}% confidence`;

        if (entry.captionStatus === 'pending') {
            detailModalCaptionArea.innerHTML = `<div class="tracked-item-caption pending">Describing...</div>`;
        } else if (entry.captionStatus === 'done') {
            detailModalCaptionArea.innerHTML = `<div class="tracked-item-caption">${escapeHtml(entry.caption)}</div>`;
        } else if (entry.captionStatus === 'error') {
            detailModalCaptionArea.innerHTML = `<div class="tracked-item-caption error">${escapeHtml(entry.caption) || 'Failed'}</div><button class="describe-btn" id="modalDescribeBtn">Retry</button>`;
        } else {
            detailModalCaptionArea.innerHTML = `<button class="describe-btn" id="modalDescribeBtn">Describe</button>`;
        }

        const btn = document.getElementById('modalDescribeBtn');
        if (btn) btn.addEventListener('click', () => requestCaption(entry.id));
    }

    function openModal(id) {
        openModalId = id;
        renderModalContent();
        detailModalBackdrop.classList.remove('hidden');
    }

    function closeModal() {
        openModalId = null;
        detailModalBackdrop.classList.add('hidden');
    }

    // Click outside the modal content (i.e. directly on the backdrop) closes it.
    detailModalBackdrop.addEventListener('click', (event) => {
        if (event.target === detailModalBackdrop) closeModal();
    });
    detailModalClose.addEventListener('click', closeModal);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && openModalId) closeModal();
    });

    // --- Prompt editing: applies to future "Describe" clicks, server-side
    // and shared across any connected viewers (see traffic_caption_prompt).
    editPromptBtn.addEventListener('click', () => {
        promptEditTextarea.value = currentPrompt;
        promptEditArea.classList.remove('hidden');
    });
    promptCancelBtn.addEventListener('click', () => {
        promptEditArea.classList.add('hidden');
    });
    promptResetBtn.addEventListener('click', () => {
        promptEditTextarea.value = defaultPrompt;
    });
    promptSaveBtn.addEventListener('click', () => {
        const newPrompt = promptEditTextarea.value.trim();
        if (newPrompt && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'update_prompt', prompt: newPrompt }));
        }
        promptEditArea.classList.add('hidden');
    });

    function connectWebSocket() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${window.location.host}/api/traffic/ws`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'new_detections') {
                for (const obj of data.objects) {
                    history.unshift(obj);
                    totalDetectionCount++;
                }
                if (history.length > MAX_HISTORY) {
                    history = history.slice(0, MAX_HISTORY);
                }
                renderHistory();
            } else if (data.type === 'caption_status') {
                const entry = history.find((h) => h.id === data.id);
                if (entry) {
                    entry.captionStatus = data.status; // 'pending' | 'done' | 'error'
                    entry.caption = data.caption;
                    renderHistory();
                    if (openModalId === data.id) renderModalContent();
                }
            } else if (data.type === 'status') {
                currentPrompt = data.prompt || '';
                defaultPrompt = data.default_prompt || '';
            } else if (data.type === 'prompt_updated') {
                currentPrompt = data.prompt || '';
                if (!promptEditArea.classList.contains('hidden')) {
                    promptEditTextarea.value = currentPrompt;
                }
            }
        };

        ws.onerror = (err) => console.error('Traffic WebSocket error:', err);
        ws.onclose = () => console.log('Traffic WebSocket closed');
    }

    function cleanup() {
        if (ws) {
            ws.onclose = null; // avoid triggering another reconnect from the close we're causing
            ws.close();
            ws = null;
        }
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
    }

    // Booth-demo robustness: any unexpected drop (ICE failure, network blip,
    // server restart) should self-heal without someone needing to reload the
    // page - same pattern as attemptAutoReconnect() in the main demo's app.js.
    function attemptAutoReconnect() {
        if (reconnectTimer) return;

        // Exponential backoff, capped - a connection that keeps failing
        // immediately (e.g. a flaky VPN interfering with ICE) should back
        // off instead of hammering the server every few seconds, which was
        // observed to overload it badly enough to crash under rapid churn.
        const delay = Math.min(
            RECONNECT_BASE_DELAY_MS * 2 ** reconnectAttempts,
            RECONNECT_MAX_DELAY_MS
        );
        reconnectAttempts++;

        updateStatus('Reconnecting...', 'processing');
        loadingOverlay.textContent = `Reconnecting... (retry ${reconnectAttempts}, waiting ${Math.round(delay / 1000)}s)`;
        loadingOverlay.classList.remove('hidden');

        reconnectTimer = setTimeout(async () => {
            reconnectTimer = null;
            isReconnecting = true;
            cleanup();
            try {
                await start();
            } catch (e) {
                console.log('Auto-reconnect attempt failed, will retry:', e);
            } finally {
                isReconnecting = false;
            }
            if (!peerConnection || peerConnection.iceConnectionState !== 'connected') {
                attemptAutoReconnect();
            }
        }, delay);
    }

    async function start() {
        try {
            updateStatus('Connecting...', 'processing');

            peerConnection = new RTCPeerConnection({ iceServers: [] });

            peerConnection.ontrack = (event) => {
                if (event.track.kind === 'video') {
                    videoElement.srcObject = event.streams[0];
                    videoElement.play().catch((err) => console.error('Error playing video:', err));
                    updateStatus('Streaming', 'connected');
                    loadingOverlay.classList.add('hidden');
                }
            };

            peerConnection.oniceconnectionstatechange = () => {
                switch (peerConnection.iceConnectionState) {
                    case 'connected':
                        updateStatus('Streaming', 'connected');
                        // Only reset backoff after staying up a while - a
                        // connection that connects then immediately drops
                        // again (e.g. the flaky-VPN case) should keep
                        // backing off, not reset to hammering every cycle.
                        clearTimeout(connectionStableTimer);
                        connectionStableTimer = setTimeout(() => {
                            reconnectAttempts = 0;
                        }, 10000);
                        break;
                    case 'disconnected':
                    case 'failed':
                    case 'closed':
                        clearTimeout(connectionStableTimer);
                        updateStatus('Disconnected', 'disconnected');
                        if (!isReconnecting && !userInitiatedStop) {
                            attemptAutoReconnect();
                        }
                        break;
                }
            };

            peerConnection.addTransceiver('video', { direction: 'recvonly' });
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            await new Promise((resolve) => {
                if (peerConnection.iceGatheringState === 'complete') {
                    resolve();
                    return;
                }
                const checkState = () => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        peerConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                peerConnection.addEventListener('icegatheringstatechange', checkState);
                setTimeout(() => {
                    peerConnection.removeEventListener('icegatheringstatechange', checkState);
                    resolve();
                }, 5000);
            });

            const response = await fetch('/api/traffic/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: peerConnection.localDescription.sdp,
                    type: peerConnection.localDescription.type,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to start traffic demo');
            }

            const answer = await response.json();
            await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

            connectWebSocket();
        } catch (error) {
            console.error('Error starting traffic demo:', error);
            updateStatus('Error', 'disconnected');
            loadingOverlay.textContent = 'Connection failed, retrying...';
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }
            if (!isReconnecting && !userInitiatedStop) {
                attemptAutoReconnect();
            }
        }
    }

    function stop() {
        userInitiatedStop = true;
        clearTimeout(reconnectTimer);
        clearTimeout(connectionStableTimer);
        reconnectTimer = null;
        cleanup();

        videoElement.srcObject = null;
        frameCount = 0;
        fpsCounter.textContent = '-- FPS';
        updateStatus('Stopped', 'disconnected');
        loadingOverlay.textContent = 'Stopped';
        loadingOverlay.classList.remove('hidden');

        toggleBtn.textContent = 'Start';
        toggleBtn.classList.add('is-stopped');
    }

    function resume() {
        userInitiatedStop = false;
        reconnectAttempts = 0;
        toggleBtn.textContent = 'Stop';
        toggleBtn.classList.remove('is-stopped');
        start();
    }

    toggleBtn.addEventListener('click', () => {
        if (userInitiatedStop) {
            resume();
        } else {
            stop();
        }
    });

    // Play/pause freezes the actual video SOURCE server-side (see
    // VideoFileTrack.pause()/resume() and traffic_websocket_handler), not
    // just the local <video> element - a live WebRTC stream keeps advancing
    // in the background regardless of whether the browser renders it, so a
    // client-only pause would "resume" showing whatever is live *now*, not
    // where you paused. Also pausing the element locally so the frozen
    // frame renders immediately rather than waiting on the next held frame.
    playPauseBtn.addEventListener('click', () => {
        const wsSend = (type) => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type }));
        };
        if (videoElement.paused) {
            videoElement.play();
            wsSend('resume');
        } else {
            videoElement.pause();
            wsSend('pause');
        }
    });
    videoElement.addEventListener('play', () => {
        playPauseBtn.innerHTML = '&#10074;&#10074;';
        playPauseBtn.title = 'Pause';
    });
    videoElement.addEventListener('pause', () => {
        playPauseBtn.innerHTML = '&#9654;';
        playPauseBtn.title = 'Play';
    });

    start();
})();
