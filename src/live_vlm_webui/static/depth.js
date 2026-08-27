// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Depth Camera Test page - Color rides the existing WebRTC + VLM pipeline
// (same /offer endpoint as webcam/RTSP/file, just with {orbbec: true}).
// Depth/IR/telemetry come over a dedicated /ws/orbbec WebSocket instead of
// more WebRTC tracks - see server.py's orbbec_ws() docstring for why.

(() => {
    const sessionId = 'depth-' + Math.random().toString(36).slice(2, 10);

    const colorVideo = document.getElementById('colorVideo');
    const depthImage = document.getElementById('depthImage');
    const irImage = document.getElementById('irImage');
    const statusBar = document.getElementById('statusBar');
    const startStopBtn = document.getElementById('startStopBtn');
    const startStopLabel = document.getElementById('startStopLabel');
    const vlmPrompt = document.getElementById('vlmPrompt');
    const vlmText = document.getElementById('vlmText');
    const vlmHistory = document.getElementById('vlmHistory');
    const vlmScroll = document.getElementById('depthVlmScroll');
    const vlmModelName = document.getElementById('vlmModelName');
    const vlmMetrics = document.getElementById('vlmMetrics');

    const promptPreset = document.getElementById('promptPreset');
    const promptText = document.getElementById('promptText');
    const answerLength = document.getElementById('answerLength');
    const maxTokens = document.getElementById('maxTokens');

    let peerConnection = null;
    let mainSocket = null;
    let orbbecSocket = null;
    let running = false;

    function setStatus(text, kind) {
        statusBar.textContent = text;
        statusBar.className = 'depth-status-bar' + (kind ? ` ${kind}` : '');
    }

    function markHasContent(bodyId, hasContent) {
        document.getElementById(bodyId).classList.toggle('has-content', hasContent);
    }

    // -- Prompt editor: same protocol/behavior as the main app's Prompt Editor
    // panel (app.js) - length instruction baked in at send time only, never
    // rewriting the textarea. Kept in sync deliberately; see app.js if this
    // logic changes there. --
    const ANSWER_LENGTH_INSTRUCTIONS = {
        '1-2': 'Answer in 1-2 sentences.',
        '3-4': 'Answer in 3-4 sentences.',
        '4-5': 'Answer in 4-5 sentences.',
        'none': '',
    };

    function buildFinalPrompt(rawPrompt) {
        const instruction = ANSWER_LENGTH_INSTRUCTIONS[answerLength.value] || '';
        if (!instruction) return rawPrompt;
        const separator = /[.!?]\s*$/.test(rawPrompt) ? ' ' : '. ';
        return rawPrompt + separator + instruction;
    }

    function applyAnswerLengthConstraintFor(selectedOption) {
        const skip = selectedOption?.dataset.skipLength === 'true';
        answerLength.disabled = skip;
        if (skip) {
            answerLength.dataset.previousValue = answerLength.value;
            answerLength.value = 'none';
        } else if (answerLength.dataset.previousValue) {
            answerLength.value = answerLength.dataset.previousValue;
            delete answerLength.dataset.previousValue;
        }
    }

    function applyPromptSettings() {
        const trimmedPrompt = promptText.value.trim();
        if (!trimmedPrompt || !mainSocket || mainSocket.readyState !== WebSocket.OPEN) return;

        const finalPrompt = buildFinalPrompt(trimmedPrompt);
        mainSocket.send(JSON.stringify({
            type: 'update_prompt',
            prompt: finalPrompt,
            max_tokens: parseInt(maxTokens.value) || 256,
        }));
        vlmPrompt.textContent = finalPrompt;
    }

    function flash(el) {
        el.classList.add('applied');
        setTimeout(() => el.classList.remove('applied'), 600);
    }

    promptPreset.addEventListener('change', (e) => {
        if (!e.target.value) return;
        promptText.value = e.target.value;
        applyAnswerLengthConstraintFor(e.target.selectedOptions[0]);
        applyPromptSettings();
        flash(promptText);
    });

    promptText.addEventListener('blur', () => {
        if (answerLength.disabled && promptText.value.trim() !== promptPreset.value) {
            applyAnswerLengthConstraintFor(null);
        }
        applyPromptSettings();
        flash(promptText);
    });

    answerLength.addEventListener('change', () => {
        applyPromptSettings();
        flash(answerLength);
    });

    maxTokens.addEventListener('change', () => {
        applyPromptSettings();
        flash(maxTokens);
    });

    // Model ids come back vendor-prefixed (e.g. "qualcomm/Qwen3-VL-4B-Instruct")
    // because that's how GenieX's catalog namespaces them - strip for display
    // only, same as displayModelName() in the main app's app.js.
    function displayModelName(modelId) {
        return modelId.includes('/') ? modelId.split('/').pop() : modelId;
    }

    // Answer scrollback - keeps the last couple of answers readable instead of
    // hard-replacing them, same behavior as the main app (see app.js
    // pushAnswerToHistory). Newest answer stays in #vlmText at the top;
    // outgoing answers drop into #vlmHistory below it.
    const MAX_ANSWER_HISTORY = 2;
    let lastVlmText = '';

    function pushAnswerToHistory(promptStr, text) {
        if (!text) return;
        const entry = document.createElement('div');
        entry.className = 'depth-vlm-history-entry';

        const promptEl = document.createElement('div');
        promptEl.className = 'depth-vlm-history-prompt';
        promptEl.textContent = promptStr || '';

        const textEl = document.createElement('div');
        textEl.className = 'depth-vlm-history-text';
        textEl.textContent = text;

        entry.appendChild(promptEl);
        entry.appendChild(textEl);
        vlmHistory.prepend(entry);

        while (vlmHistory.children.length > MAX_ANSWER_HISTORY) {
            vlmHistory.removeChild(vlmHistory.lastElementChild);
        }
    }

    function clearAnswerHistory() {
        vlmHistory.innerHTML = '';
        lastVlmText = '';
    }

    function connectMainSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        mainSocket = new WebSocket(`${protocol}//${window.location.host}/ws?session_id=${encodeURIComponent(sessionId)}`);

        mainSocket.onopen = () => applyPromptSettings();

        mainSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'vlm_response' && data.text) {
                if (data.text !== lastVlmText) {
                    if (lastVlmText) pushAnswerToHistory(vlmPrompt.textContent, lastVlmText);
                    vlmText.textContent = data.text;
                    lastVlmText = data.text;
                    vlmScroll.scrollTo({ top: 0, behavior: 'smooth' });

                    // Pop-in + glow on the new answer
                    vlmText.classList.remove('new-message', 'with-glow');
                    void vlmText.offsetWidth; // force reflow so the animation restarts
                    vlmText.classList.add('new-message', 'with-glow');
                    setTimeout(() => vlmText.classList.remove('new-message', 'with-glow'), 400);
                }
                if (data.metrics) {
                    vlmMetrics.style.display = 'flex';
                    document.getElementById('vlmLatency').textContent = Math.round(data.metrics.last_latency_ms);
                    document.getElementById('vlmAvgLatency').textContent = Math.round(data.metrics.avg_latency_ms);
                    document.getElementById('vlmCount').textContent = data.metrics.total_inferences;
                }
            } else if (data.type === 'server_config' && data.model) {
                vlmModelName.textContent = displayModelName(data.model);
            } else if (data.type === 'gpu_stats' && data.stats) {
                updateSystemTelemetry(data.stats);
            }
        };
    }

    // Same ring-gauge math as the main app's app.js (setRingProgress) - a
    // fixed circumference (2*pi*42, r=42) SVG stroke-dashoffset trick.
    const RING_CIRCUMFERENCE = 263.9;
    function setRingProgress(elementId, percent) {
        const ring = document.getElementById(elementId);
        if (!ring) return;
        const clamped = Math.max(0, Math.min(100, percent || 0));
        ring.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - clamped / 100);
        ring.classList.toggle('high', clamped >= 85);
    }

    function themeColor(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }

    // Multi-series line chart (CPU/iGPU/NPU temp history) - ported from the
    // main app's drawMultiSparkline in app.js. series: [{data, color}, ...]
    function drawMultiSparkline(canvas, series) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        const allValues = [];
        series.forEach((s) => {
            if (s.data) s.data.forEach((v) => { if (v !== null && v !== undefined) allValues.push(v); });
        });
        if (allValues.length === 0) return;

        const min = Math.min(...allValues);
        const max = Math.max(...allValues, min + 1);
        const range = (max - min) || 1;
        const pad = height * 0.1;

        series.forEach((s) => {
            if (!s.data || s.data.length === 0) return;
            const step = width / (s.data.length - 1 || 1);
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            let started = false;
            s.data.forEach((value, index) => {
                if (value === null || value === undefined) return;
                const x = index * step;
                const y = height - pad - ((value - min) / range) * (height - 2 * pad);
                if (!started) {
                    ctx.moveTo(x, y);
                    started = true;
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.stroke();
        });
    }

    function updateSystemTelemetry(stats) {
        const fmtTemp = (t) => (t === null || t === undefined) ? 'N/A' : `${t.toFixed(1)}°C`;

        const cpuPercent = stats.cpu_percent || 0;
        document.getElementById('telCpuPercent').textContent = `${cpuPercent.toFixed(1)}%`;
        setRingProgress('cpuRing', cpuPercent);

        const ramUsed = stats.ram_used_gb || 0;
        const ramTotal = stats.ram_total_gb || 0;
        const ramPercent = stats.ram_percent || 0;
        document.getElementById('telRam').innerHTML =
            `${ramUsed.toFixed(1)}<span class="stat-value-denominator">/${ramTotal.toFixed(1)}GB</span>`;
        setRingProgress('ramRing', ramPercent);

        document.getElementById('telCpuTemp').textContent = fmtTemp(stats.cpu_temp_c);
        document.getElementById('telGpuTemp').textContent = fmtTemp(stats.gpu_temp_c);
        document.getElementById('telNpuTemp').textContent = fmtTemp(stats.npu_temp_c);

        if (stats.history) {
            const chart = document.getElementById('depthThermalChart');
            if (chart) {
                // Canvas has its own internal drawing-buffer resolution (300x150
                // default) separate from its CSS display size - without matching
                // them, the browser stretches the buffer non-uniformly and every
                // gentle wiggle in the data reads as a sharp jagged spike.
                const width = chart.offsetWidth || chart.parentElement.offsetWidth || 300;
                const height = chart.offsetHeight || 70;
                if (chart.width !== width || chart.height !== height) {
                    chart.width = width;
                    chart.height = height;
                }
                drawMultiSparkline(chart, [
                    { data: stats.history.cpu_temp, color: themeColor('--thermal-cpu') },
                    { data: stats.history.gpu_temp, color: themeColor('--thermal-igpu') },
                    { data: stats.history.npu_temp, color: themeColor('--thermal-npu') },
                ]);
            }
        }

        // Host OS/hardware identity, for the "Local System Stats" drawer tab
        document.getElementById('sysBoard').textContent = stats.qc_board_name || '--';
        document.getElementById('sysSoc').textContent = stats.qc_soc_name || '--';
        document.getElementById('sysCpu').textContent = stats.qc_cpu_model || stats.cpu_model || '--';
        document.getElementById('sysGpu').textContent = stats.qc_gpu_name || stats.gpu_name || '--';
        document.getElementById('sysNpu').textContent = stats.qc_npu_name || '--';
        document.getElementById('sysHostname').textContent = stats.hostname || '--';
        document.getElementById('sysOs').textContent = stats.os_pretty || '--';
        document.getElementById('sysKernel').textContent = stats.kernel_version || '--';
    }

    function connectOrbbecSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        orbbecSocket = new WebSocket(`${protocol}//${window.location.host}/ws/orbbec`);

        orbbecSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'error') {
                setStatus(`Orbbec error: ${data.message}`, 'error');
                return;
            }
            if (data.type !== 'orbbec_frame') return;

            if (data.depth) {
                depthImage.src = `data:image/jpeg;base64,${data.depth}`;
                markHasContent('depthPanel', true);
            }
            if (data.ir) {
                irImage.src = `data:image/jpeg;base64,${data.ir}`;
                markHasContent('irPanel', true);
            }
            if (data.telemetry) {
                updateTelemetry(data.telemetry);
            }
        };

        orbbecSocket.onerror = () => setStatus('Orbbec WebSocket error', 'error');
    }

    function updateTelemetry(telemetry) {
        const device = telemetry.device || {};
        document.getElementById('telDeviceName').textContent = device.name || '--';
        document.getElementById('telDeviceSn').textContent = device.serial_number || '--';
        document.getElementById('telDeviceFw').textContent = device.firmware_version || '--';
        document.getElementById('telDeviceConn').textContent = device.connection_type || '--';

        const accel = telemetry.accel;
        document.getElementById('telAccelX').textContent = accel ? accel.x.toFixed(3) : '--';
        document.getElementById('telAccelY').textContent = accel ? accel.y.toFixed(3) : '--';
        document.getElementById('telAccelZ').textContent = accel ? accel.z.toFixed(3) : '--';

        const gyro = telemetry.gyro;
        document.getElementById('telGyroX').textContent = gyro ? gyro.x.toFixed(4) : '--';
        document.getElementById('telGyroY').textContent = gyro ? gyro.y.toFixed(4) : '--';
        document.getElementById('telGyroZ').textContent = gyro ? gyro.z.toFixed(4) : '--';

        const temp = (accel && accel.temperature) ?? (gyro && gyro.temperature);
        document.getElementById('telTemp').textContent = temp != null ? `${temp.toFixed(1)} °C` : '--';
    }

    async function connectColor() {
        peerConnection = new RTCPeerConnection({ iceServers: [] });

        peerConnection.ontrack = (event) => {
            if (event.track.kind === 'video') {
                colorVideo.srcObject = event.streams[0];
                colorVideo.play().catch((err) => console.error('Error playing color video:', err));
                markHasContent('colorPanel', true);
            }
        };

        peerConnection.oniceconnectionstatechange = () => {
            if (['failed', 'disconnected', 'closed'].includes(peerConnection.iceConnectionState) && running) {
                setStatus('Color stream disconnected', 'error');
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
            const check = () => {
                if (peerConnection.iceGatheringState === 'complete') {
                    peerConnection.removeEventListener('icegatheringstatechange', check);
                    resolve();
                }
            };
            peerConnection.addEventListener('icegatheringstatechange', check);
            setTimeout(resolve, 5000);
        });

        const response = await fetch('/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: peerConnection.localDescription.sdp,
                type: peerConnection.localDescription.type,
                orbbec: true,
                session_id: sessionId,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to connect to Orbbec camera');
        }

        const answer = await response.json();
        await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    }

    async function start() {
        setStatus('Connecting…', 'processing');
        startStopBtn.disabled = true;
        try {
            connectMainSocket();
            connectOrbbecSocket();
            await connectColor();

            running = true;
            setStatus('Streaming', 'connected');
            startStopLabel.textContent = 'Stop';
            startStopBtn.classList.add('active');
            startStopBtn.querySelector('.lucide').setAttribute('data-lucide', 'square');
            lucide.createIcons();
        } catch (error) {
            console.error('Failed to start depth camera test:', error);
            setStatus(`Error: ${error.message}`, 'error');
            stop();
        } finally {
            startStopBtn.disabled = false;
        }
    }

    function stop() {
        running = false;
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
        if (mainSocket) {
            mainSocket.close();
            mainSocket = null;
        }
        if (orbbecSocket) {
            orbbecSocket.close();
            orbbecSocket = null;
        }
        colorVideo.srcObject = null;
        markHasContent('colorPanel', false);
        markHasContent('depthPanel', false);
        markHasContent('irPanel', false);
        clearAnswerHistory();
        vlmText.textContent = 'Waiting for first answer…';
        vlmMetrics.style.display = 'none';

        setStatus('Not connected');
        startStopLabel.textContent = 'Start';
        startStopBtn.classList.remove('active');
        const icon = startStopBtn.querySelector('.lucide');
        if (icon) {
            icon.setAttribute('data-lucide', 'play');
            lucide.createIcons();
        }
    }

    startStopBtn.addEventListener('click', () => {
        if (running) {
            stop();
        } else {
            start();
        }
    });

    // IR panel doubles as an inline system-telemetry view (CPU/RAM rings +
    // thermal chart) - "like the normal view" but embedded in the grid
    // instead of a separate card, since screen space here is tighter.
    const irSystemToggle = document.getElementById('irSystemToggle');
    const irPanelTitle = document.getElementById('irPanelTitle');
    const irPanelBody = document.getElementById('irPanelBody');
    const systemPanelBody = document.getElementById('systemPanelBody');
    let showingSystemView = false;

    irSystemToggle.addEventListener('click', () => {
        showingSystemView = !showingSystemView;
        irPanelBody.style.display = showingSystemView ? 'none' : '';
        systemPanelBody.style.display = showingSystemView ? '' : 'none';
        irPanelTitle.textContent = showingSystemView ? 'System' : 'IR';
        irSystemToggle.classList.toggle('active', showingSystemView);
    });

    // Fullscreen - CSS-only overlay (see .depth-panel.fullscreen), same
    // approach as the main app's video card. Works on any of the three
    // feed panels; only one can be fullscreen at a time.
    function toggleFullscreen(panel, btn) {
        const isFullscreen = panel.classList.toggle('fullscreen');
        const icon = btn.querySelector('.lucide');
        if (icon) {
            icon.setAttribute('data-lucide', isFullscreen ? 'minimize' : 'maximize');
            lucide.createIcons();
        }
    }

    function exitAllFullscreen() {
        document.querySelectorAll('.depth-panel.fullscreen').forEach((panel) => {
            panel.classList.remove('fullscreen');
            const btn = panel.querySelector('.depth-panel-toggle[id$="FullscreenBtn"]');
            const icon = btn?.querySelector('.lucide');
            if (icon) icon.setAttribute('data-lucide', 'maximize');
        });
        lucide.createIcons();
    }

    [
        ['colorFullscreenBtn', 'colorPanel'],
        ['depthFullscreenBtn', 'depthPanel'],
        ['irFullscreenBtn', 'irPanel'],
    ].forEach(([btnId, panelId]) => {
        const btn = document.getElementById(btnId);
        const panel = document.getElementById(panelId);
        btn.addEventListener('click', () => toggleFullscreen(panel, btn));
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.querySelector('.depth-panel.fullscreen')) {
            exitAllFullscreen();
        }
    });

    // Side panel drawer (Local System Stats / Prompt tabs). The IR panel's
    // inline toggle is just a quick-glance CPU/RAM/thermal widget; this
    // drawer has the fuller picture (host OS/hardware identity + camera IMU).
    const telemetryBtn = document.getElementById('telemetryBtn');
    const promptBtn = document.getElementById('promptBtn');
    const telemetryDrawer = document.getElementById('telemetryDrawer');
    const telemetryOverlay = document.getElementById('telemetryOverlay');
    const telemetryCloseBtn = document.getElementById('telemetryCloseBtn');
    const tabButtons = {
        stats: document.getElementById('tabBtnStats'),
        prompt: document.getElementById('tabBtnPrompt'),
    };
    const tabContents = {
        stats: document.getElementById('tabStats'),
        prompt: document.getElementById('tabPrompt'),
    };

    function showTab(name) {
        for (const key of Object.keys(tabButtons)) {
            tabButtons[key].classList.toggle('active', key === name);
            tabContents[key].style.display = key === name ? '' : 'none';
        }
    }

    function openDrawer(tab) {
        telemetryDrawer.classList.add('open');
        telemetryOverlay.classList.add('open');
        if (tab) showTab(tab);
    }
    function closeDrawer() {
        telemetryDrawer.classList.remove('open');
        telemetryOverlay.classList.remove('open');
    }

    telemetryBtn.addEventListener('click', () => openDrawer('stats'));
    promptBtn.addEventListener('click', () => openDrawer('prompt'));
    telemetryCloseBtn.addEventListener('click', closeDrawer);
    telemetryOverlay.addEventListener('click', closeDrawer);
    tabButtons.stats.addEventListener('click', () => showTab('stats'));
    tabButtons.prompt.addEventListener('click', () => showTab('prompt'));

    lucide.createIcons();
})();
