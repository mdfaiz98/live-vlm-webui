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
    const trackedPanel = document.getElementById('trackedPanel');
    const videoWrap = document.getElementById('videoWrap');
    const trackedCount = document.getElementById('trackedCount');
    const toggleBtn = document.getElementById('toggleBtn');
    const detectionToggleBtn = document.getElementById('detectionToggleBtn');
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

    // Detection starts OFF on every (re)connect - see DetectionVideoTrack's
    // detection_enabled in detection_processor.py. The plain video plays
    // immediately; detection (boxes + panel population) only turns on once
    // this button is clicked, for a live "watch it turn on" demo moment.
    let detectionEnabled = false;

    function setDetectionButtonState(enabled) {
        detectionEnabled = enabled;
        detectionToggleBtn.textContent = enabled ? 'Stop Detection' : 'Start Detection';
        detectionToggleBtn.classList.toggle('is-running', enabled);
    }

    detectionToggleBtn.addEventListener('click', () => {
        const next = !detectionEnabled;
        setDetectionButtonState(next);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'toggle_detection', enabled: next }));
        }
    });

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

    // ---- Telemetry column (gpu_stats from the server's system monitor) ----
    // Same data as the main demo's System Stats card; on Qualcomm boards the
    // monitor adds GPU busy % + clock (driver), NPU busy % (detector's own
    // HTP time + GenieX VLM request time, capped at 100%) - see QualcommMonitor.
    const RING_CIRCUMFERENCE = 263.9; // 2 * pi * r42
    const TELEMETRY_MIN_INTERVAL_MS = 500; // server sends at 4 Hz; this is plenty for a dashboard
    let lastTelemetryRender = 0;

    function themeColor(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el && el.textContent !== text) el.textContent = text;
    }

    function setRing(id, percent) {
        const ring = document.getElementById(id);
        if (!ring) return;
        const clamped = Math.max(0, Math.min(100, percent || 0));
        ring.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - clamped / 100);
        ring.classList.toggle('high', clamped >= 85);
    }

    function sizeCanvas(canvas) {
        const width = canvas.offsetWidth || canvas.parentElement.offsetWidth || 260;
        const height = canvas.offsetHeight || 40;
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
    }

    // Multi-series line chart; fixedRange {min, max} (e.g. 0-100 for
    // utilization), otherwise auto-scaled across all series. fill adds a
    // translucent area under single-series charts.
    function drawChart(canvas, series, fixedRange = null, fill = false) {
        sizeCanvas(canvas);
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        ctx.clearRect(0, 0, width, height);

        const all = [];
        series.forEach((s) => (s.data || []).forEach((v) => { if (v != null) all.push(v); }));
        if (all.length === 0) return;
        const min = fixedRange ? fixedRange.min : Math.min(...all);
        const max = fixedRange ? fixedRange.max : Math.max(...all, min + 1);
        const range = (max - min) || 1;
        const pad = height * 0.1;

        series.forEach((s) => {
            if (!s.data || s.data.length === 0) return;
            const step = width / (s.data.length - 1 || 1);
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            let started = false;
            let lastX = 0;
            s.data.forEach((v, i) => {
                if (v == null) return; // gap for missing readings
                const x = i * step;
                const y = height - pad - ((v - min) / range) * (height - 2 * pad);
                if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
                lastX = x;
            });
            ctx.stroke();
            if (fill && started) {
                ctx.lineTo(lastX, height);
                ctx.lineTo(0, height);
                ctx.closePath();
                ctx.fillStyle = s.color + '20';
                ctx.fill();
            }
        });
    }

    function renderTelemetry(stats) {
        const now = Date.now();
        if (now - lastTelemetryRender < TELEMETRY_MIN_INTERVAL_MS) return;
        lastTelemetryRender = now;

        // Header - same three-line format as the main demo's System Stats card:
        // board + SoC (hostname) with CPU / GPU + NPU / OS + kernel
        const header = document.getElementById('systemInfoHeader');
        let headerHtml;
        if (stats.qc_board_name) {
            const soc = stats.qc_soc_name ? ` (${escapeHtml(stats.qc_soc_name)})` : '';
            const cpuModel = stats.qc_cpu_model ? ` with ${escapeHtml(stats.qc_cpu_model)}` : '';
            const npu = stats.qc_npu_name ? ` &middot; ${escapeHtml(stats.qc_npu_name)}` : '';
            const os = stats.os_pretty
                ? `<br>${escapeHtml(stats.os_pretty)}${stats.kernel_version ? ' (' + escapeHtml(stats.kernel_version) + ')' : ''}`
                : '';
            headerHtml = `<b>${escapeHtml(stats.qc_board_name)}</b>${soc} (<code>${escapeHtml(stats.hostname || '')}</code>)${cpuModel}<br>` +
                `${escapeHtml(stats.qc_gpu_name || 'Adreno GPU')}${npu}${os}`;
        } else {
            headerHtml = `<code>${escapeHtml(stats.hostname || 'System')}</code><br>with ${escapeHtml(stats.cpu_model || '')}`;
        }
        if (header.innerHTML !== headerHtml) header.innerHTML = headerHtml;

        const cpu = stats.cpu_percent || 0;
        setText('cpuUtil', `${cpu.toFixed(1)}%`);
        setRing('cpuRing', cpu);
        const ramHtml = `${(stats.ram_used_gb || 0).toFixed(1)}<span class="stat-value-denominator">/${(stats.ram_total_gb || 0).toFixed(1)}GB</span>`;
        const ramEl = document.getElementById('ramUsage');
        if (ramEl.innerHTML !== ramHtml) ramEl.innerHTML = ramHtml;
        setRing('ramRing', stats.ram_percent || 0);

        const accelCard = document.getElementById('accelCard');
        const hasAccel = stats.gpu_freq_mhz != null || stats.npu_percent != null;
        accelCard.hidden = !hasAccel;
        if (hasAccel) {
            const pctText = (v) => (v == null ? 'N/A' : `${v.toFixed(0)}%`);
            setText('gpuUtil', pctText(stats.gpu_percent));
            setText('npuUtil', pctText(stats.npu_percent));
            setText('gpuFreq', stats.gpu_freq_mhz != null ? `${stats.gpu_freq_mhz.toFixed(0)} MHz` : '-- MHz');
            setRing('gpuRing', stats.gpu_percent);
            setRing('npuRing', stats.npu_percent);
        }

        const fmtTemp = (t) => (t == null ? 'N/A' : `${t.toFixed(1)}°C`);
        setText('cpuTempValue', fmtTemp(stats.cpu_temp_c));
        setText('gpuTempValue', fmtTemp(stats.gpu_temp_c));
        setText('npuTempValue', fmtTemp(stats.npu_temp_c));

        const h = stats.history;
        if (!h) return;
        drawChart(document.getElementById('cpuSparkline'),
            [{ data: h.cpu_util, color: themeColor('--thermal-cpu') }], { min: 0, max: 100 }, true);
        drawChart(document.getElementById('ramSparkline'),
            [{ data: h.ram_used, color: themeColor('--accent-color-2') }], null, true);
        if (hasAccel) {
            drawChart(document.getElementById('accelChart'), [
                { data: h.gpu_util, color: themeColor('--thermal-igpu') },
                { data: h.npu_util, color: themeColor('--thermal-npu') },
            ], { min: 0, max: 100 });
        }
        drawChart(document.getElementById('thermalChart'), [
            { data: h.cpu_temp, color: themeColor('--thermal-cpu') },
            { data: h.gpu_temp, color: themeColor('--thermal-igpu') },
            { data: h.npu_temp, color: themeColor('--thermal-npu') },
        ]);
    }

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
            } else if (data.type === 'settings_updated') {
                if (data.detection_model) {
                    detectionModelSelect.value = data.detection_model;
                    showDetectionModelName();
                    detectionModelSelect.disabled = false;
                    setHint(detectionModelHint, 'Runs on the Hexagon NPU \u00b7 switches live');
                }
                if (data.vlm_model) vlmModelSelect.value = data.vlm_model;
            } else if (data.type === 'settings_error') {
                detectionModelSelect.disabled = false;
                setHint(detectionModelHint, data.text, true);
                loadSettings(); // put the select back on the model that's actually active
            } else if (data.type === 'gpu_stats') {
                renderTelemetry(data.stats);
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
            setDetectionButtonState(false); // every fresh connection starts with detection off

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
                    video: savedVideo(),
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

    // Keep the Detected Vehicles panel exactly as tall as the video (the list
    // scrolls inside it), through window resizes and full screen.
    new ResizeObserver(() => {
        trackedPanel.style.height = `${videoWrap.offsetHeight}px`;
        trackedPanel.style.maxHeight = 'none';
    }).observe(videoWrap);

    // Page full screen: the whole UI without browser chrome. Leaving it gives
    // the normal browser window back. Ported from the drone detection demo.
    const pageFullscreenBtn = document.getElementById('pageFullscreenBtn');
    const pageFullscreenIcon = document.getElementById('pageFullscreenIcon');
    const ICON_MAXIMIZE = '<path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/>';
    const ICON_MINIMIZE = '<path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M3 21l7-7"/>';
    pageFullscreenBtn.addEventListener('click', () => {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen().catch(() => {});
        }
    });
    document.addEventListener('fullscreenchange', () => {
        const on = !!document.fullscreenElement;
        pageFullscreenIcon.innerHTML = on ? ICON_MINIMIZE : ICON_MAXIMIZE;
        pageFullscreenBtn.title = on ? 'Exit full screen (Esc)' : 'Full screen (Esc to exit)';
    });
    // /traffic?fullscreen=1 enters full screen on load where the browser allows
    // it without a click (e.g. a kiosk profile); elsewhere it's just ignored.
    if (new URLSearchParams(location.search).get('fullscreen') === '1') {
        document.documentElement.requestFullscreen().catch(() => {});
    }

    // ---- Theme: same Auto -> Light -> Dark cycle and saved preference as the main demo ----
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const themeText = document.getElementById('themeText');

    function applyTheme(theme) {
        const light = theme === 'light' ||
            (theme === 'auto' && window.matchMedia('(prefers-color-scheme: light)').matches);
        document.body.classList.toggle('light-theme', light);
        const icon = { light: 'sun', dark: 'moon', auto: 'monitor' }[theme];
        themeIcon.innerHTML = `<i data-lucide="${icon}"></i>`;
        themeText.textContent = theme === 'auto' ? 'Auto' : theme === 'light' ? 'Light' : 'Dark';
        if (window.lucide) lucide.createIcons();
        document.getElementById('systemProductImage').src =
            light ? '/images/m48-workstation-256px-blk.png' : '/images/m48-workstation-256px-wht.png';
        lastTelemetryRender = 0; // redraw charts in the new theme's colours on the next update
    }

    function savedTheme() {
        let t = null;
        try { t = localStorage.getItem('theme'); } catch (e) { /* storage blocked */ }
        return t === 'light' || t === 'dark' ? t : 'auto';
    }

    themeToggle.addEventListener('click', () => {
        const next = { auto: 'light', light: 'dark', dark: 'auto' }[savedTheme()];
        try {
            if (next === 'auto') localStorage.removeItem('theme');
            else localStorage.setItem('theme', next);
        } catch (e) { /* storage blocked - theme still applies for this visit */ }
        applyTheme(next);
    });
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
        if (savedTheme() === 'auto') applyTheme('auto');
    });
    applyTheme(savedTheme());

    // ---- Settings drawer: video source, detection model, VLM model ----
    const settingsDrawer = document.getElementById('settingsDrawer');
    const settingsBackdrop = document.getElementById('settingsBackdrop');
    const videoSelect = document.getElementById('videoSelect');
    const detectionModelSelect = document.getElementById('detectionModelSelect');
    const detectionModelHint = document.getElementById('detectionModelHint');
    const vlmModelSelect = document.getElementById('vlmModelSelect');
    const VIDEO_KEY = 'trafficVideo';
    const videoUploadInput = document.getElementById('videoUploadInput');
    const videoUploadStatus = document.getElementById('videoUploadStatus');

    function setHint(el, text, isError = false) {
        el.textContent = text;
        el.classList.toggle('error', isError);
    }

    // The video the next connection plays. Kept for this tab only (reconnects,
    // switching), so every fresh page load starts on the server default - the
    // highway video. null = server default.
    function savedVideo() {
        try { return sessionStorage.getItem(VIDEO_KEY); } catch (e) { return null; }
    }

    function fillSelect(select, options, value) {
        select.innerHTML = options.map((o) => `<option value="${escapeHtml(o.id)}">${escapeHtml(o.name)}</option>`).join('');
        if (value != null && options.some((o) => o.id === value)) select.value = value;
    }

    // "Vehicles detected on-device (<model>, Hexagon NPU)" under the video
    function showDetectionModelName() {
        const opt = detectionModelSelect.selectedOptions[0];
        if (opt) document.getElementById('detectionModelName').textContent = opt.textContent;
    }

    async function loadSettings(selectVideo = null) {
        try {
            const res = await fetch('/api/traffic/settings');
            const s = await res.json();
            fillSelect(videoSelect, s.videos, selectVideo || savedVideo() || s.default_video);
            fillSelect(detectionModelSelect, s.detection_models, s.detection_model);
            showDetectionModelName();
            fillSelect(vlmModelSelect, s.vlm_models, s.vlm_model);
        } catch (e) {
            console.error('Could not load settings:', e);
        }
    }

    function openSettings() {
        loadSettings();
        settingsBackdrop.hidden = false;
        settingsDrawer.classList.add('open');
    }

    function closeSettings() {
        settingsDrawer.classList.remove('open');
        settingsBackdrop.hidden = true;
    }

    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('settingsCloseBtn').addEventListener('click', closeSettings);
    settingsBackdrop.addEventListener('click', closeSettings);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && settingsDrawer.classList.contains('open')) closeSettings();
    });

    // Play this video: save it, then reconnect with it
    document.getElementById('applySourceBtn').addEventListener('click', () => {
        try { sessionStorage.setItem(VIDEO_KEY, videoSelect.value); } catch (e) { /* storage blocked */ }
        // A fresh connection starts with detection off; clear the old video's vehicles
        history = [];
        totalDetectionCount = 0;
        renderHistory();
        closeSettings();
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
        reconnectAttempts = 0;
        cleanup();
        videoElement.srcObject = null;
        loadingOverlay.textContent = 'Switching video...';
        loadingOverlay.classList.remove('hidden');
        userInitiatedStop = false;
        toggleBtn.textContent = 'Stop';
        toggleBtn.classList.remove('is-stopped');
        start();
    });

    videoUploadInput.addEventListener('change', async () => {
        const file = videoUploadInput.files[0];
        if (!file) return;
        setHint(videoUploadStatus, `Uploading ${file.name}...`);
        try {
            const form = new FormData();
            form.append('file', file);
            const res = await fetch('/api/traffic/upload', { method: 'POST', body: form });
            const out = await res.json();
            if (!res.ok) throw new Error(out.error || 'Upload failed');
            await loadSettings(out.id);
            setHint(videoUploadStatus, `Added ${out.name} - click Play this video`);
        } catch (e) {
            setHint(videoUploadStatus, e.message, true);
        }
        videoUploadInput.value = '';
    });

    detectionModelSelect.addEventListener('change', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            detectionModelSelect.disabled = true;
            setHint(detectionModelHint, 'Loading model on the NPU...');
            ws.send(JSON.stringify({ type: 'set_detection_model', id: detectionModelSelect.value }));
        }
    });

    vlmModelSelect.addEventListener('change', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'set_vlm_model', id: vlmModelSelect.value }));
        }
    });

    loadSettings();

    start();
})();
