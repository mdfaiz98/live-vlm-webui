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

        // Initialize Lucide icons immediately when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => lucide.createIcons());
        } else {
            lucide.createIcons();
        }

        // --- Motion helpers -------------------------------------------------
        // Small shared wrappers around window.Motion (vendored, see /vendor/motion.min.js).
        // Every call degrades to an instant, no-op state change if Motion isn't
        // loaded or the user has requested reduced motion - never a hard requirement.
        function prefersReducedMotion() {
            return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        }

        function motionAnimate(element, keyframes, options) {
            if (!window.Motion || prefersReducedMotion()) {
                return Promise.resolve();
            }
            return Promise.resolve(window.Motion.animate(element, keyframes, options));
        }

        function updateMetricText(element, newText) {
            if (!element || element.textContent === newText) return;
            element.textContent = newText;
        }

        function updateMetricHTML(element, newHTML) {
            if (!element || element.innerHTML === newHTML) return;
            element.innerHTML = newHTML;
        }

        // Elements
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const themeText = document.getElementById('themeText');
        const videoElement = document.getElementById('videoElement');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const connectionStatus = document.getElementById('connectionStatus');
        const resultText = document.getElementById('resultText');
        const currentPrompt = document.getElementById('currentPrompt');
        const metricsInline = document.getElementById('metricsInline');
        const markdownToggle = document.getElementById('markdownToggle');
        const markdownIcon = document.getElementById('markdownIcon');
        const markdownText = document.getElementById('markdownText');
        const copyButton = document.getElementById('copyButton');
        const latencyValue = document.getElementById('latencyValue');
        const avgLatencyValue = document.getElementById('avgLatencyValue');
        const countValue = document.getElementById('countValue');
        const promptPreset = document.getElementById('promptPreset');
        const promptText = document.getElementById('promptText');
        const maxTokens = document.getElementById('maxTokens');
        const modelSelect = document.getElementById('modelSelect');
        const refreshModelsBtn = document.getElementById('refreshModelsBtn');
        const processEvery = document.getElementById('processEvery');
        const maxLatency = document.getElementById('maxLatency');
        const mirrorBtn = document.getElementById('mirrorBtn');

        let peerConnection = null;
        let localStream = null;
        let websocket = null;
        let fadeTimeout = null;
        let lastText = '';
        // Multi-session: one ID per page/tab; sent in WebSocket and /offer so this tab gets only its VLM output
        const sessionId = crypto.randomUUID ? crypto.randomUUID() : 'tab-' + Date.now() + '-' + Math.random().toString(36).slice(2);
        let isAnalysisRunning = false;
        let selectedCameraId = null;
        let rtspSessionId = 'default';  // For RTSP mode
        // Enable markdown by default (user can disable if they prefer raw text)
        let markdownEnabled = localStorage.getItem('markdownEnabled') !== 'false';

        // Enumerate and list available cameras
        async function enumerateCameras() {
            try {
                // Request permission first to get device labels
                const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
                tempStream.getTracks().forEach(track => track.stop());

                // Now enumerate with labels
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');

                const cameraSelect = document.getElementById('cameraSelect');
                cameraSelect.innerHTML = '';

                if (videoDevices.length === 0) {
                    cameraSelect.innerHTML = '<option value="">No cameras found</option>';
                    return;
                }

                videoDevices.forEach((device, index) => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${index + 1}`;
                    cameraSelect.appendChild(option);
                });

                // Select first camera by default
                selectedCameraId = videoDevices[0].deviceId;
                console.log(`Found ${videoDevices.length} camera(s)`);
            } catch (err) {
                console.error('Error enumerating cameras:', err);
                document.getElementById('cameraSelect').innerHTML = '<option value="">Error detecting cameras</option>';
            }
        }

        // Handle camera selection change
        document.getElementById('cameraSelect').addEventListener('change', async (e) => {
            selectedCameraId = e.target.value;
            console.log('Selected camera:', selectedCameraId);

            // If already running, switch camera
            if (isAnalysisRunning && localStream) {
                try {
                    // Stop current stream
                    localStream.getTracks().forEach(track => track.stop());

                    // Get new stream with selected camera
                localStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                            deviceId: { exact: selectedCameraId },
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                        }
                    });

                    // Update video element
                    videoElement.srcObject = localStream;

                    // Replace track in peer connection
                    const videoTrack = localStream.getVideoTracks()[0];
                    const sender = peerConnection.getSenders().find(s => s.track && s.track.kind === 'video');
                    if (sender) {
                        await sender.replaceTrack(videoTrack);
                        console.log('Camera switched successfully');
                    }
                } catch (err) {
                    console.error('Error switching camera:', err);
                    alert('Failed to switch camera: ' + err.message);
                }
            }

            // Trigger flash animation
            const cameraSelect = document.getElementById('cameraSelect');
            cameraSelect.classList.add('applied');
            setTimeout(() => {
                cameraSelect.classList.remove('applied');
            }, 600);
        });

        // Also trigger flash on blur (when dropdown closes)
        document.getElementById('cameraSelect').addEventListener('blur', () => {
            const cameraSelect = document.getElementById('cameraSelect');
            if (cameraSelect.value) {
                // Trigger flash animation
                cameraSelect.classList.add('applied');
                setTimeout(() => {
                    cameraSelect.classList.remove('applied');
                }, 600);
            }
        });

        // Tooltip positioning (for fixed position tooltips)
        document.querySelectorAll('.tooltip-wrapper').forEach(wrapper => {
            const icon = wrapper.querySelector('.tooltip-icon');
            const tooltip = wrapper.querySelector('.tooltip-text');

            if (icon && tooltip) {
                wrapper.addEventListener('mouseenter', () => {
                    const rect = icon.getBoundingClientRect();
                    tooltip.style.left = `${rect.left}px`;
                    tooltip.style.top = `${rect.bottom + 8}px`;
                });
            }
        });

        // Handle input source tabs (Webcam vs RTSP vs Video File)
        document.querySelectorAll('.input-source-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const source = tab.getAttribute('data-source');
                const webcamControls = document.getElementById('webcamControls');
                const rtspControls = document.getElementById('rtspControls');
                const videoFileControls = document.getElementById('videoFileControls');
                const rtspBetaWarning = document.getElementById('rtspBetaWarning');
                const startBtn = document.getElementById('startBtn');

                // Update active tab
                document.querySelectorAll('.input-source-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // Show/hide controls
                webcamControls.style.display = source === 'webcam' ? 'block' : 'none';
                rtspControls.style.display = source === 'rtsp' ? 'block' : 'none';
                videoFileControls.style.display = source === 'videofile' ? 'block' : 'none';
                rtspBetaWarning.style.display = source === 'rtsp' ? 'flex' : 'none';
            });
        });

        // Clear RTSP status message when user types
        const rtspUrlInput = document.getElementById('rtspUrl');
        rtspUrlInput.addEventListener('input', () => {
            const statusDiv = document.getElementById('rtspStatus');
            statusDiv.style.display = 'none';
        });

        // Green flash animation when RTSP URL is set (same as API Base URL)
        rtspUrlInput.addEventListener('blur', () => {
            if (rtspUrlInput.value.trim()) {
                rtspUrlInput.classList.add('applied');
                setTimeout(() => {
                    rtspUrlInput.classList.remove('applied');
                }, 600);
            }
        });

        // Test RTSP connection
        document.getElementById('testRtspBtn').addEventListener('click', async () => {
            const rtspUrl = document.getElementById('rtspUrl').value.trim();
            const statusDiv = document.getElementById('rtspStatus');
            const testBtn = document.getElementById('testRtspBtn');

            if (!rtspUrl) {
                statusDiv.textContent = '⚠️ Please enter an RTSP URL';
                statusDiv.style.display = 'block';
                statusDiv.style.color = 'var(--error-color)';
                return;
            }

            testBtn.disabled = true;
            statusDiv.textContent = '🔄 Testing connection...';
            statusDiv.style.display = 'block';
            statusDiv.style.color = 'var(--success-color)';

            try {
                const response = await fetch('/api/rtsp/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        rtsp_url: rtspUrl,
                        session_id: 'test-' + Date.now()
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    const info = data.stream_info;
                    statusDiv.innerHTML = `✅ Connected!<br>${info.codec} ${info.width}x${info.height} @${info.fps}fps`;
                    statusDiv.style.color = 'var(--success-color)';

                    // Stop the test connection after 2 seconds
                    setTimeout(async () => {
                        await fetch('/api/rtsp/stop', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: data.session_id })
                        });
                    }, 2000);
                } else {
                    statusDiv.textContent = '❌ ' + (data.error || 'Connection failed');
                    statusDiv.style.color = 'var(--error-color)';
                }
            } catch (err) {
                console.error('RTSP test error:', err);
                statusDiv.textContent = '❌ Connection failed: ' + err.message;
                statusDiv.style.color = 'var(--error-color)';
            } finally {
                testBtn.disabled = false;
            }
        });

        // Tooltip positioning - dynamically position tooltip to avoid clipping
        document.querySelectorAll('.tooltip-wrapper').forEach(wrapper => {
            const icon = wrapper.querySelector('.tooltip-icon');
            const tooltip = wrapper.querySelector('.tooltip-text');

            if (icon && tooltip) {
                icon.addEventListener('mouseenter', () => {
                    const rect = icon.getBoundingClientRect();
                    const tooltipWidth = 380; // Match CSS width

                    // Position below the icon
                    let left = rect.left - 10; // Slight offset to left
                    let top = rect.bottom + 8;

                    // Ensure tooltip doesn't go off right edge of viewport
                    if (left + tooltipWidth > window.innerWidth - 20) {
                        left = window.innerWidth - tooltipWidth - 20;
                    }

                    // Ensure tooltip doesn't go off left edge
                    if (left < 20) {
                        left = 20;
                    }

                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                });
            }
        });

        // Markdown rendering function
        function renderMarkdown(text) {
            if (!text) return '';
            // Check if libraries are loaded
            if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
                console.warn('Markdown libraries not loaded, falling back to plain text');
                return text;
            }
            try {
                // Configure marked options
                marked.setOptions({
                    breaks: true,  // Convert line breaks to <br>
                    gfm: true,     // GitHub Flavored Markdown
                });
                // Parse markdown and sanitize HTML
                const html = marked.parse(text);
                return DOMPurify.sanitize(html, {
                    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'code', 'pre', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr', 'a'],
                    ALLOWED_ATTR: ['href', 'title'],
                });
            } catch (e) {
                console.error('Error rendering markdown:', e);
                return text; // Fallback to plain text on error
            }
        }

        // Update markdown toggle UI
        function updateMarkdownToggleUI() {
            const contentDiv = document.getElementById('resultTextContent');
            if (markdownEnabled) {
                markdownIcon.innerHTML = '<i data-lucide="code"></i>';
                markdownText.textContent = 'Markdown';
                if (contentDiv) {
                    contentDiv.classList.add('markdown-rendered');
                } else {
                    resultText.classList.add('markdown-rendered');
                }
            } else {
                markdownIcon.innerHTML = '<i data-lucide="file-text"></i>';
                markdownText.textContent = 'Plain Text';
                if (contentDiv) {
                    contentDiv.classList.remove('markdown-rendered');
                } else {
                    resultText.classList.remove('markdown-rendered');
                }
            }
            lucide.createIcons();
        }

        // Markdown toggle handler
        markdownToggle.addEventListener('click', () => {
            markdownEnabled = !markdownEnabled;
            localStorage.setItem('markdownEnabled', markdownEnabled.toString());
            updateMarkdownToggleUI();

            // Re-render current text with new mode
            if (lastText) {
                updateResultText(lastText);
            }
        });

        // Initialize markdown toggle state
        updateMarkdownToggleUI();

        // Copy to clipboard functionality
        copyButton.addEventListener('click', async () => {
            const contentDiv = document.getElementById('resultTextContent');
            let textToCopy = '';

            if (contentDiv) {
                // If markdown is enabled, copy the raw text (not HTML)
                if (markdownEnabled) {
                    // Get the raw text by reading from lastText or extracting from content
                    textToCopy = lastText || contentDiv.innerText || contentDiv.textContent;
                } else {
                    textToCopy = contentDiv.textContent || contentDiv.innerText;
                }
            } else {
                textToCopy = resultText.textContent || resultText.innerText;
            }

            if (!textToCopy || textToCopy.trim() === '') {
                return; // Nothing to copy
            }

            try {
                await navigator.clipboard.writeText(textToCopy);

                // Visual feedback
                const originalHTML = copyButton.innerHTML;

                copyButton.classList.add('copied');
                copyButton.innerHTML = '<i data-lucide="check"></i>';
                lucide.createIcons();

                // Reset after 0.8 seconds
                setTimeout(() => {
                    copyButton.classList.remove('copied');
                    copyButton.innerHTML = originalHTML;
                    lucide.createIcons();
                }, 800);
            } catch (err) {
                console.error('Failed to copy text:', err);
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = textToCopy;
                textArea.style.position = 'fixed';
                textArea.style.opacity = '0';
                document.body.appendChild(textArea);
                textArea.select();
                try {
                    document.execCommand('copy');
                    copyButton.classList.add('copied');
                    setTimeout(() => copyButton.classList.remove('copied'), 500);
                } catch (e) {
                    console.error('Fallback copy failed:', e);
                }
                document.body.removeChild(textArea);
            }
        });

        // Function to update result text (handles both markdown and plain text)
        function updateResultText(text) {
            const contentDiv = document.getElementById('resultTextContent');
            if (!contentDiv) {
                // Fallback if content div doesn't exist yet
                if (markdownEnabled) {
                    resultText.innerHTML = renderMarkdown(text);
                } else {
                    resultText.textContent = text;
                }
                return;
            }
            // Update only the content div, preserving the button
            if (markdownEnabled) {
                contentDiv.innerHTML = renderMarkdown(text);
            } else {
                contentDiv.textContent = text;
            }

            // Auto-sync to fullscreen overlay if in fullscreen mode
            const videoCard = document.getElementById('videoCard');
            if (videoCard && videoCard.classList.contains('fullscreen')) {
                syncVlmToFullscreen();
            }
        }

        // Enumerate cameras on page load
        enumerateCameras();

        // Theme management: Honor OS preference with manual override support
        function applyTheme(theme) {
            // theme can be 'light', 'dark', or 'auto'
            if (theme === 'light') {
                document.body.classList.add('light-theme');
                themeIcon.innerHTML = '<i data-lucide="sun"></i>';
                themeText.textContent = 'Light';
            } else if (theme === 'dark') {
                document.body.classList.remove('light-theme');
                themeIcon.innerHTML = '<i data-lucide="moon"></i>';
                themeText.textContent = 'Dark';
            } else { // 'auto' - follow OS preference
                const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
                if (prefersLight) {
                    document.body.classList.add('light-theme');
                } else {
                    document.body.classList.remove('light-theme');
                }
                // Always show monitor icon for AUTO mode
                themeIcon.innerHTML = '<i data-lucide="monitor"></i>';
                themeText.textContent = 'Auto';
            }
            lucide.createIcons();

            // Update product image for theme-aware icons (Mac/PC workstation icons)
            if (window.lastSystemStats) {
                const productImg = document.getElementById('systemProductImage');
                const imageSrc = getSystemProductImage(window.lastSystemStats);
                if (imageSrc) {
                    productImg.src = imageSrc;
                    // Also update background image
                    const systemStatsCard = document.getElementById('systemStatsCard');
                    systemStatsCard.style.setProperty('--system-bg-image', `url('${imageSrc}')`);
                }
            }
        }

        // Theme Toggle: cycles through Auto -> Light -> Dark -> Auto
        themeToggle.addEventListener('click', () => {
            const currentTheme = localStorage.getItem('theme') || 'auto';
            let nextTheme;

            if (currentTheme === 'auto' || !currentTheme) {
                nextTheme = 'light';
            } else if (currentTheme === 'light') {
                nextTheme = 'dark';
            } else { // dark
                nextTheme = 'auto';
            }

            // Only save manual overrides (light/dark) to localStorage
            // Don't save 'auto' - always check OS preference when in auto mode
            if (nextTheme === 'auto') {
                localStorage.removeItem('theme');
            } else {
                localStorage.setItem('theme', nextTheme);
            }
            applyTheme(nextTheme);
        });

        // Load saved theme or detect OS preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light' || savedTheme === 'dark') {
            // User has manually set a preference - use it
            applyTheme(savedTheme);
        } else {
            // No manual preference saved - follow OS preference
            applyTheme('auto');
        }

        // Listen for OS preference changes when in 'auto' mode
        const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: light)');
        colorSchemeQuery.addEventListener('change', (e) => {
            const currentTheme = localStorage.getItem('theme');
            // Only update if user hasn't manually set a preference
            if (!currentTheme || currentTheme === 'auto') {
                applyTheme('auto');
            }
        });

        // Load saved colorful UI accents setting (default: disabled for clean two-tone system)
        const colorfulFocusToggle = document.getElementById('colorfulFocusToggle');
        const colorfulFocusSaved = localStorage.getItem('colorfulFocus');
        if (colorfulFocusSaved === 'true') {
            document.body.classList.add('colorful-focus');
            colorfulFocusToggle.checked = true;
        } else {
            // Default: neutral white/gray icons and focus glows with cyan flash
            document.body.classList.remove('colorful-focus');
            colorfulFocusToggle.checked = false;
        }

        // Settings Modal
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const settingsClose = document.getElementById('settingsClose');
        const popInToggle = document.getElementById('popInToggle');
        const glowToggle = document.getElementById('glowToggle');
        const fadeToggle = document.getElementById('fadeToggle');
        const overlayPosition = document.getElementById('overlayPosition');
        const layoutOrder = document.getElementById('layoutOrder');
        const videoOverlay = document.getElementById('videoOverlay');
        const gpuUpdateInterval = document.getElementById('gpuUpdateInterval');

        // Settings state
        const settings = {
            popIn: localStorage.getItem('popIn') !== 'false',
            glow: localStorage.getItem('glow') !== 'false',
            fade: localStorage.getItem('fade') !== 'false',
            overlayPosition: localStorage.getItem('overlayPosition') || 'none',
            layoutOrder: localStorage.getItem('layoutOrder') || 'vlm-first',
            gpuUpdateInterval: parseFloat(localStorage.getItem('gpuUpdateInterval')) || 0.25
        };

        // Load saved settings
        popInToggle.checked = settings.popIn;
        glowToggle.checked = settings.glow;
        fadeToggle.checked = settings.fade;
        overlayPosition.value = settings.overlayPosition;
        layoutOrder.value = settings.layoutOrder;
        gpuUpdateInterval.value = settings.gpuUpdateInterval;

        // Apply overlay position
        function applyOverlayPosition(position) {
            videoOverlay.classList.remove('show', 'top', 'bottom');
            const resultText = document.getElementById('resultText');

            if (position !== 'none') {
                // Overlay enabled - show on video, hide response balloon (keep prompt visible)
                videoOverlay.classList.add('show', position);
                resultText.style.display = 'none';
            } else {
                // Overlay disabled - hide from video, show response balloon
                resultText.style.display = 'block';
            }
        }
        applyOverlayPosition(settings.overlayPosition);

        // Apply layout order
        function applyLayoutOrder(order) {
            const mainContent = document.querySelector('.main-content');
            const videoCard = document.querySelector('.video-card');
            const resultCard = document.querySelector('.result-card');

            if (order === 'vlm-first') {
                // VLM Output Info at top, Camera below (default)
                mainContent.insertBefore(resultCard, videoCard);
            } else {
                // Camera at top, VLM Output Info below
                mainContent.insertBefore(videoCard, resultCard);
            }
        }
        applyLayoutOrder(settings.layoutOrder);

        // Settings modal open/close - animated via Motion (falls back to an
        // instant class toggle if Motion isn't loaded or reduced-motion is on).
        const settingsDialog = settingsModal.querySelector('.settings-dialog');

        function openSettingsModal() {
            settingsModal.classList.add('show');
            motionAnimate(settingsModal, { opacity: [0, 1] }, { duration: 0.15, easing: 'ease-out' });
            motionAnimate(settingsDialog, { opacity: [0, 1], scale: [0.95, 1] },
                { duration: 0.22, easing: [0.16, 1, 0.3, 1] });
        }

        function closeSettingsModal() {
            Promise.all([
                motionAnimate(settingsModal, { opacity: [1, 0] }, { duration: 0.15, easing: 'ease-in' }),
                motionAnimate(settingsDialog, { opacity: [1, 0], scale: [1, 0.97] }, { duration: 0.15, easing: 'ease-in' }),
            ]).then(() => settingsModal.classList.remove('show'));
        }

        settingsBtn.addEventListener('click', openSettingsModal);
        settingsClose.addEventListener('click', closeSettingsModal);

        // Close modal when clicking outside
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) {
                closeSettingsModal();
            }
        });

        // Handle settings changes
        popInToggle.addEventListener('change', (e) => {
            settings.popIn = e.target.checked;
            localStorage.setItem('popIn', settings.popIn);
        });

        glowToggle.addEventListener('change', (e) => {
            settings.glow = e.target.checked;
            localStorage.setItem('glow', settings.glow);
        });

        fadeToggle.addEventListener('change', (e) => {
            settings.fade = e.target.checked;
            localStorage.setItem('fade', settings.fade);
        });

        colorfulFocusToggle.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            localStorage.setItem('colorfulFocus', enabled);
            if (enabled) {
                document.body.classList.add('colorful-focus');
            } else {
                document.body.classList.remove('colorful-focus');
            }
            // Force icon re-render to apply new colors
            lucide.createIcons();
        });

        overlayPosition.addEventListener('change', (e) => {
            settings.overlayPosition = e.target.value;
            localStorage.setItem('overlayPosition', settings.overlayPosition);
            applyOverlayPosition(settings.overlayPosition);
        });

        layoutOrder.addEventListener('change', (e) => {
            settings.layoutOrder = e.target.value;
            localStorage.setItem('layoutOrder', settings.layoutOrder);
            applyLayoutOrder(settings.layoutOrder);
        });

        gpuUpdateInterval.addEventListener('change', (e) => {
            const newInterval = parseFloat(e.target.value);
            if (!isNaN(newInterval) && newInterval >= 0.1 && newInterval <= 5.0) {
                settings.gpuUpdateInterval = newInterval;
                localStorage.setItem('gpuUpdateInterval', settings.gpuUpdateInterval);
                console.log(`🔄 GPU graph update interval changed to ${newInterval}s (updates every ${newInterval}s)`);
                console.log(`📊 Note: Server still sends stats every 0.25s, but graphs will only redraw at the configured rate`);
                // Reset lastGPUUpdate to allow immediate update with new interval
                lastGPUUpdate = 0;
            } else {
                console.warn('Invalid GPU update interval, keeping previous value');
                e.target.value = settings.gpuUpdateInterval;
            }
        });

        // Panel Toggle
        function togglePanel(panelId) {
            const content = document.getElementById(panelId);
            const toggle = document.getElementById(panelId + 'Toggle');
            content.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }

        // Control drawer (right-side panel) collapse/expand
        let sidebarDrawerCollapsed = false;

        function setSidebarDrawerCollapsed(collapsed) {
            const drawer = document.getElementById('controlDrawer');
            if (!drawer) return;
            sidebarDrawerCollapsed = collapsed;
            drawer.classList.toggle('drawer-collapsed', collapsed);
            const latchIcon = document.getElementById('drawerLatchIcon');
            const latch = document.getElementById('drawerLatch');
            if (latchIcon) {
                latchIcon.setAttribute('data-lucide', collapsed ? 'chevron-left' : 'chevron-right');
            }
            if (latch) {
                latch.title = collapsed ? 'Expand panel' : 'Collapse panel';
            }
            if (window.lucide) {
                lucide.createIcons();
            }
        }

        function toggleSidebarDrawer() {
            setSidebarDrawerCollapsed(!sidebarDrawerCollapsed);
        }

        // Used by the collapsed-state icon rail: re-expand the drawer and make
        // sure the requested panel is open (not left collapsed from before)
        function expandDrawerToPanel(panelId) {
            setSidebarDrawerCollapsed(false);
            const allPanelIds = ['vlmConfig', 'cameraConfig', 'promptEditor'];
            allPanelIds.forEach(id => {
                const content = document.getElementById(id);
                const toggle = document.getElementById(id + 'Toggle');
                if (!content) return;
                if (id === panelId) {
                    // Expand the requested panel
                    content.classList.remove('collapsed');
                    if (toggle) toggle.classList.remove('collapsed');
                } else {
                    // Collapse the others so only one panel is open at a time
                    content.classList.add('collapsed');
                    if (toggle) toggle.classList.add('collapsed');
                }
            });
        }

        // Fullscreen Toggle
        function toggleFullscreen() {
            const videoCard = document.getElementById('videoCard');
            const fullscreenIcon = document.getElementById('fullscreenIcon');

            videoCard.classList.toggle('fullscreen');

            // Update icon: maximize when normal, minimize when fullscreen
            if (videoCard.classList.contains('fullscreen')) {
                fullscreenIcon.setAttribute('data-lucide', 'minimize');
                // Sync current VLM output to fullscreen overlay
                syncVlmToFullscreen();
            } else {
                fullscreenIcon.setAttribute('data-lucide', 'maximize');
            }
            lucide.createIcons();
        }

        // Sync VLM output to fullscreen overlay
        function syncVlmToFullscreen() {
            const mainVlmContent = document.getElementById('resultTextContent');
            const fullscreenVlmContent = document.getElementById('fullscreenVlmContent');
            const fullscreenVlmMetrics = document.getElementById('fullscreenVlmMetrics');

            if (mainVlmContent && fullscreenVlmContent) {
                fullscreenVlmContent.innerHTML = mainVlmContent.innerHTML || 'Ready';
            }

            // Add metrics if available
            const latency = document.getElementById('latencyValue')?.textContent;
            const count = document.getElementById('countValue')?.textContent;
            if (latency && count) {
                fullscreenVlmMetrics.innerHTML = `<span>Latency: ${latency}ms</span><span>Count: ${count}</span>`;
            }
        }

        // ESC key to exit fullscreen
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const videoCard = document.getElementById('videoCard');
                if (videoCard.classList.contains('fullscreen')) {
                    toggleFullscreen();
                }
            }
        });

        // API Key Field Toggle
        function toggleApiKeyField() {
            const field = document.getElementById('apiKeyField');
            const toggle = document.getElementById('apiKeyToggle');
            field.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }

        // Check if API Base URL requires API key
        function checkApiKeyRequirement(url) {
            const apiKeyField = document.getElementById('apiKeyField');
            const apiKeyToggle = document.getElementById('apiKeyToggle');

            // Check if URL is a remote service (not localhost or 127.0.0.1)
            const isLocal = url.includes('localhost') || url.includes('127.0.0.1') || url.includes('0.0.0.0');
            const isRemote = !isLocal;

            if (isRemote) {
                // Show API key field for remote services
                apiKeyField.classList.remove('collapsed');
                apiKeyToggle.classList.remove('collapsed');
                console.log('Remote API detected - API Key field expanded');
            } else {
                // Hide API key field for local services
                apiKeyField.classList.add('collapsed');
                apiKeyToggle.classList.add('collapsed');
                console.log('Local API detected - API Key field collapsed');
            }
        }

        // Status Update
        function updateStatus(message, state) {
            connectionStatus.textContent = message;
            connectionStatus.className = `status-badge ${state}`;
        }

        // Detect Local VLM Services
        async function detectServices() {
            try {
                const response = await fetch('/detect-services');
                const data = await response.json();

                if (data.default) {
                    const service = data.default;
                    console.log('Detected service:', service.name);

                    // Update API Base URL
                    apiBaseUrl.value = service.url;

                    // Check if API key is required
                    checkApiKeyRequirement(service.url);
                    updateSystemStatsVisibility(service.url);

                    // Update hint text
                    const hintDiv = apiBaseUrl.nextElementSibling;
                    if (data.detected.length > 1) {
                        const serviceNames = data.detected.map(s => s.name).join(', ');
                        hintDiv.textContent = `Detected: ${serviceNames}`;
                    } else if (service.name === 'NVIDIA API Catalog') {
                        hintDiv.textContent = 'No local VLM services found. Using NVIDIA API Catalog (requires API key from build.nvidia.com)';
                    }
                }
            } catch (error) {
                console.error('Error detecting services:', error);
                // Default to showing API key field on error
                checkApiKeyRequirement('https://');
            }
        }

        // Fetch Models
        async function fetchModels() {
            try {
                modelSelect.innerHTML = '<option value="">Loading models...</option>';

                // Get current API settings from UI
                const currentApiBase = apiBaseUrl.value.trim();
                const currentApiKey = apiKey.value.trim();

                // Build query params
                const params = new URLSearchParams();
                if (currentApiBase) {
                    params.append('api_base', currentApiBase);
                }
                if (currentApiKey) {
                    params.append('api_key', currentApiKey);
                }

                const url = `/models${params.toString() ? '?' + params.toString() : ''}`;
                const response = await fetch(url);
                const data = await response.json();

                if (data.models && data.models.length > 0) {
                    modelSelect.innerHTML = '';
                    let currentModel = null;
                    let autoSelectedModel = null;

                    data.models.forEach((model, index) => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.id;
                        if (model.current) {
                            option.selected = true;
                            currentModel = model.id;
                        }
                        modelSelect.appendChild(option);
                    });

                    // If no current model, auto-select first model and apply
                    if (!currentModel && data.models.length > 0) {
                        autoSelectedModel = data.models[0].id;
                        modelSelect.value = autoSelectedModel;
                        currentModel = autoSelectedModel;

                        // Auto-apply the new model
                        console.log(`Auto-selected model: ${autoSelectedModel}`);
                        applyApiSettings({ showFeedback: false });
                    }

                    // Update model name in VLM output header
                    if (currentModel) {
                        document.getElementById('modelName').textContent = currentModel;
                    }
                } else {
                    modelSelect.innerHTML = '<option value="">No models available</option>';
                }
            } catch (error) {
                console.error('Error fetching models:', error);
                modelSelect.innerHTML = '<option value="">Error loading models</option>';
            }
        }

        // Handle model change
        modelSelect.addEventListener('change', (e) => {
            const newModel = e.target.value;
            if (!newModel) return;

            // Update model name in VLM output header
            document.getElementById('modelName').textContent = newModel;

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                // Send model, API base, and API key together
                websocket.send(JSON.stringify({
                    type: 'update_model',
                    model: newModel,
                    api_base: apiBaseUrl.value.trim(),
                    api_key: apiKey.value.trim()
                }));
                updateStatus('Model configured', 'connected');
            }

            // Trigger flash animation
            modelSelect.classList.add('applied');
            setTimeout(() => {
                modelSelect.classList.remove('applied');
            }, 600);
        });

        // Also trigger flash on blur (when dropdown closes)
        modelSelect.addEventListener('blur', () => {
            const currentModel = modelSelect.value;
            if (currentModel) {
                // Trigger flash animation
                modelSelect.classList.add('applied');
                setTimeout(() => {
                    modelSelect.classList.remove('applied');
                }, 600);
            }
        });

        // Helper function to apply prompt settings
        function applyPromptSettings() {
            const newPrompt = promptText.value.trim();
            const tokens = parseInt(maxTokens.value) || 512;

            if (!newPrompt) {
                return; // Silently skip if empty
            }

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_prompt',
                    prompt: newPrompt,
                    max_tokens: tokens
                }));

                // Update the displayed prompt (chat bubble style)
                const promptPreview = newPrompt.length > 150 ? newPrompt.substring(0, 150) + '...' : newPrompt;
                currentPrompt.textContent = `Prompt: ${promptPreview}`;
            }
        }

        // Handle preset selection
        promptPreset.addEventListener('change', (e) => {
            if (e.target.value) {
                promptText.value = e.target.value;
                applyPromptSettings();

                // Trigger flash animation to show prompt was applied
                promptText.classList.add('applied');
                setTimeout(() => {
                    promptText.classList.remove('applied');
                }, 600); // Match animation duration
            }
        });

        // Auto-apply prompt on blur (when user finishes editing)
        promptText.addEventListener('blur', () => {
            applyPromptSettings();

            // Trigger flash animation to show prompt was applied
            promptText.classList.add('applied');
            setTimeout(() => {
                promptText.classList.remove('applied');
            }, 600); // Match animation duration
        });

        // Auto-apply when max tokens changes
        maxTokens.addEventListener('change', () => {
            applyPromptSettings();

            // Trigger flash animation to show setting was applied
            maxTokens.classList.add('applied');
            setTimeout(() => {
                maxTokens.classList.remove('applied');
            }, 600); // Match animation duration
        });

        // Also trigger flash on blur (after tabbing away or clicking out)
        maxTokens.addEventListener('blur', () => {
            applyPromptSettings();

            // Trigger flash animation
            maxTokens.classList.add('applied');
            setTimeout(() => {
                maxTokens.classList.remove('applied');
            }, 600);
        });

        refreshModelsBtn.addEventListener('click', fetchModels);

        // Helper function to apply API settings to server
        function applyApiSettings(options = {}) {
            const currentApiBase = apiBaseUrl.value.trim();
            const currentApiKey = apiKey.value.trim();
            const currentModel = modelSelect.value;

            if (!currentApiBase) {
                return; // Silently skip if no API base
            }

            if (!currentModel) {
                return; // Silently skip if no model selected yet
            }

            // Send update to server
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_model',
                    model: currentModel,
                    api_base: currentApiBase,
                    api_key: currentApiKey
                }));

                if (options.showFeedback) {
                    updateStatus('API settings updated', 'connected');
                }

                // Refresh models from new endpoint if requested
                if (options.refreshModels) {
                    fetchModels();
                }
            }
        }

        // Auto-apply API settings when API Base URL changes
        apiBaseUrl.addEventListener('blur', (e) => {
            checkApiKeyRequirement(e.target.value);
            updateSystemStatsVisibility(e.target.value);
            applyApiSettings({ refreshModels: true, showFeedback: true });

            // Trigger flash animation
            apiBaseUrl.classList.add('applied');
            setTimeout(() => {
                apiBaseUrl.classList.remove('applied');
            }, 600);
        });

        apiBaseUrl.addEventListener('change', (e) => {
            checkApiKeyRequirement(e.target.value);
            updateSystemStatsVisibility(e.target.value);
            applyApiSettings({ refreshModels: true, showFeedback: true });

            // Trigger flash animation
            apiBaseUrl.classList.add('applied');
            setTimeout(() => {
                apiBaseUrl.classList.remove('applied');
            }, 600);
        });

        // Auto-apply API settings when API Key changes (with debounce)
        let apiKeyDebounceTimer;
        apiKey.addEventListener('input', () => {
            clearTimeout(apiKeyDebounceTimer);
            apiKeyDebounceTimer = setTimeout(() => {
                applyApiSettings({ showFeedback: false });
            }, 1000); // Wait 1 second after user stops typing
        });

        // Flash animation when API Key loses focus
        apiKey.addEventListener('blur', () => {
            clearTimeout(apiKeyDebounceTimer);
            applyApiSettings({ showFeedback: false });

            // Trigger flash animation
            apiKey.classList.add('applied');
            setTimeout(() => {
                apiKey.classList.remove('applied');
            }, 600);
        });

        // API Presets Menu
        const apiPresetsBtn = document.getElementById('apiPresetsBtn');
        const apiPresetsMenu = document.getElementById('apiPresetsMenu');

        apiPresetsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            apiPresetsMenu.style.display = apiPresetsMenu.style.display === 'none' ? 'block' : 'none';
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!apiPresetsMenu.contains(e.target) && e.target !== apiPresetsBtn) {
                apiPresetsMenu.style.display = 'none';
            }
        });

        // Handle preset selection
        document.querySelectorAll('.api-preset-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const url = e.currentTarget.getAttribute('data-url');
                apiBaseUrl.value = url;
                apiPresetsMenu.style.display = 'none';

                // Trigger checks
                checkApiKeyRequirement(url);
                updateSystemStatsVisibility(url);

                // Auto-apply new API settings
                applyApiSettings({ refreshModels: true, showFeedback: true });

                // Update hint
                const hint = document.getElementById('apiBaseHint');
                const serviceName = e.currentTarget.querySelector('strong').textContent;
                hint.textContent = `Using ${serviceName}`;
            });
        });

        // Update system stats visibility based on API endpoint
        function updateSystemStatsVisibility(url) {
            const systemStatsCard = document.getElementById('systemStatsCard');
            const isLocal = url.includes('localhost') || url.includes('127.0.0.1') || url.includes('0.0.0.0');

            if (isLocal) {
                systemStatsCard.classList.remove('cloud-api');
            } else {
                systemStatsCard.classList.add('cloud-api');
            }
        }

        // Mirror video toggle
        mirrorBtn.addEventListener('click', () => {
            videoElement.classList.toggle('mirrored');
            mirrorBtn.classList.toggle('active');
        });

        // Auto-apply processing interval on change
        processEvery.addEventListener('change', () => {
            const interval = parseInt(processEvery.value) || 30;

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_processing',
                    process_every: interval
                }));
            }

            // Trigger flash animation
            processEvery.classList.add('applied');
            setTimeout(() => {
                processEvery.classList.remove('applied');
            }, 600);
        });

        // Also trigger flash on blur
        processEvery.addEventListener('blur', () => {
            const interval = parseInt(processEvery.value) || 30;

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_processing',
                    process_every: interval
                }));
            }

            // Trigger flash animation
            processEvery.classList.add('applied');
            setTimeout(() => {
                processEvery.classList.remove('applied');
            }, 600);
        });

        // Auto-apply max latency on change
        maxLatency.addEventListener('change', () => {
            const latency = parseFloat(maxLatency.value) || 1.0;

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_max_latency',
                    max_latency: latency
                }));
            }
        });

        // Debug payload toggles (Settings modal): send set_debug so server includes request/response payload in vlm_response
        function sendDebugFlags() {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
            const req = document.getElementById('debugShowRequestPayload');
            const res = document.getElementById('debugShowResponsePayload');
            if (req && res) {
                websocket.send(JSON.stringify({
                    type: 'set_debug',
                    show_request_payload: req.checked,
                    show_response_payload: res.checked
                }));
            }
        }
        const debugShowRequestPayload = document.getElementById('debugShowRequestPayload');
        const debugShowResponsePayload = document.getElementById('debugShowResponsePayload');
        const requestPayloadDebugEl = document.getElementById('requestPayloadDebug');
        const responsePayloadDebugEl = document.getElementById('responsePayloadDebug');
        if (debugShowRequestPayload) {
            debugShowRequestPayload.addEventListener('change', () => {
                sendDebugFlags();
                if (requestPayloadDebugEl) requestPayloadDebugEl.style.display = debugShowRequestPayload.checked ? 'block' : 'none';
            });
        }
        if (debugShowResponsePayload) {
            debugShowResponsePayload.addEventListener('change', () => {
                sendDebugFlags();
                if (responsePayloadDebugEl) responsePayloadDebugEl.style.display = debugShowResponsePayload.checked ? 'block' : 'none';
            });
        }

        // Read a theme color token so chart colors stay in sync with dark/light mode
        function themeColor(varName) {
            return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        }

        // Sparkline Drawing
        function drawSparkline(canvas, data, color) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            if (!data || data.length === 0) return;

            // Filter out null/undefined values and convert to numbers
            const validData = data.map(v => v === null || v === undefined ? 0 : v);
            const max = Math.max(...validData, 1);
            const step = width / (validData.length - 1 || 1);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            validData.forEach((value, index) => {
                const x = index * step;
                const y = height - (value / max) * height;

                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();

            ctx.lineTo(width, height);
            ctx.lineTo(0, height);
            ctx.closePath();
            ctx.fillStyle = color + '20';
            ctx.fill();
        }

        // Multi-series line chart (used for the Thermal card: CPU / iGPU / NPU temps)
        // series: array of { data: number[]|null, color: '#hex' }
        function drawMultiSparkline(canvas, series) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            // Determine a shared scale across all series so lines are comparable
            const allValues = [];
            series.forEach(s => {
                if (s.data) {
                    s.data.forEach(v => { if (v !== null && v !== undefined) allValues.push(v); });
                }
            });
            if (allValues.length === 0) return;

            const min = Math.min(...allValues);
            const max = Math.max(...allValues, min + 1);
            const range = (max - min) || 1;
            // Small padding so lines don't touch the very top/bottom edge
            const pad = height * 0.1;

            series.forEach(s => {
                if (!s.data || s.data.length === 0) return;
                const validData = s.data.map(v => (v === null || v === undefined) ? null : v);
                const step = width / (validData.length - 1 || 1);

                ctx.strokeStyle = s.color;
                ctx.lineWidth = 2;
                ctx.beginPath();

                let started = false;
                validData.forEach((value, index) => {
                    if (value === null) return;  // gap for missing readings
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

        // Get product image based on system stats
        function getSystemProductImage(stats) {
            const productName = stats.product_name || '';
            const boardName = stats.board_name || '';
            const platform = stats.platform || '';
            const gpuName = stats.gpu_name || '';

            // Jetson boards
            if (boardName.includes('Thor') || productName.includes('Thor')) {
                return '/images/jetson-agx-thor-devkit_256px.png';
            }
            // Check for Orin Nano first (more specific)
            if (boardName.includes('Orin Nano') || productName.includes('Orin Nano')) {
                return '/images/jetson-orin-nano-devkit_217px.png';
            }
            // Then check for general Orin (AGX Orin)
            if (boardName.includes('Orin') || productName.includes('Orin')) {
                return '/images/jetson-agx-orin-devkit_256px.png';
            }

            // DGX systems
            if (productName.includes('DGX Spark')) {
                return '/images/dgx-spark_256px.png';
            }

            // MacBook - use laptop icon (theme-aware)
            if (productName.includes('MacBook')) {
                const isLightTheme = document.body.classList.contains('light-theme');
                return isLightTheme ? '/images/m48-laptop-256px-blk.png' : '/images/m48-laptop-256px-wht.png';
            }

            // Other Mac products (Mac mini, iMac, Mac Studio, Mac Pro) - use workstation icon
            if (productName.includes('Mac')) {
                const isLightTheme = document.body.classList.contains('light-theme');
                return isLightTheme ? '/images/m48-workstation-256px-blk.png' : '/images/m48-workstation-256px-wht.png';
            }

            // PC with NVIDIA GPU - show NVIDIA GPU chip image
            if (gpuName.toLowerCase().includes('nvidia') || gpuName.toLowerCase().includes('geforce') ||
                gpuName.toLowerCase().includes('rtx') || gpuName.toLowerCase().includes('gtx') ||
                gpuName.toLowerCase().includes('quadro') || gpuName.toLowerCase().includes('tesla')) {
                return '/images/m48-gpu-chip-text-256px-grn.png';
            }

            // Generic PC - use workstation icon
            const isLightTheme = document.body.classList.contains('light-theme');
            return isLightTheme ? '/images/m48-workstation-256px-blk.png' : '/images/m48-workstation-256px-wht.png';
        }

        // GPU Stats Throttling
        let lastGPUUpdate = 0;
        let pendingGPUStats = null;
        let gpuUpdateCount = 0;

        // Update GPU Stats (throttled based on settings.gpuUpdateInterval)
        function updateGPUStats(stats) {
            // Diagnostic logging (log every 10th update to avoid console spam)
            gpuUpdateCount++;
            if (gpuUpdateCount % 10 === 0) {
                const gpuPercent = stats.gpu_percent !== null && stats.gpu_percent !== undefined ? stats.gpu_percent.toFixed(1) : 'N/A';
                console.log(`📊 GPU Update #${gpuUpdateCount}: GPU=${gpuPercent}%, VRAM=${stats.vram_used_gb?.toFixed(1)}GB, CPU=${stats.cpu_percent?.toFixed(1)}%`);
            }

            // Update system info header
            const hostname = stats.hostname || 'System';
            const cpuModel = stats.cpu_model || 'Unknown CPU';
            const gpuName = stats.gpu_name || 'Unknown GPU';
            let boardName = stats.board_name;  // e.g., "Jetson AGX Thor Developer Kit"

            // Safety check: ensure boardName is a string (not an object)
            if (boardName && typeof boardName === 'object') {
                console.warn('Board name is an object, expected string:', boardName);
                boardName = null;  // Ignore it and fall back to regular display
            }

            // Format header based on platform
            const systemInfoElem = document.getElementById('systemInfoHeader');
            const productName = stats.product_name;
            const gpuCores = stats.gpu_cores || 0;

            // Check if there's a discrete GPU (not integrated graphics)
            const hasDiscreteGPU = gpuName &&
                !gpuName.includes('N/A') &&
                !gpuName.includes('Unknown') &&
                !gpuName.includes('Intel HD') &&
                !gpuName.includes('Intel UHD') &&
                !gpuName.includes('Intel Iris') &&
                gpuName !== 'CPU';

            // Helper: Remove frequency from CPU name for compact display
            const cpuModelShort = cpuModel.replace(/\s*@\s*[\d.]+\s*[GM]Hz/gi, '').replace(/\s+/g, ' ').trim();

            if (stats.qc_board_name) {
                // Qualcomm Snapdragon/Dragonwing board (e.g. Advantech AIR-055 / QCS9075)
                // Three-line format: board+SoC+CPU (hostname), GPU/NPU, OS/kernel
                const qcSoc = stats.qc_soc_name ? ` (${stats.qc_soc_name})` : '';
                const qcCpu = stats.qc_cpu_model ? ` with ${stats.qc_cpu_model}` : '';
                const qcGpu = stats.qc_gpu_name || 'Adreno GPU';
                const qcNpu = stats.qc_npu_name ? ` &middot; ${stats.qc_npu_name}` : '';
                const osLine = stats.os_pretty
                    ? `${stats.os_pretty}${stats.kernel_version ? ' (' + stats.kernel_version + ')' : ''}`
                    : '';
                systemInfoElem.innerHTML =
                    `<b>${stats.qc_board_name}</b>${qcSoc} (<code>${hostname}</code>)${qcCpu}<br>` +
                    `${qcGpu}${qcNpu}` +
                    (osLine ? `<br>${osLine}` : '');
            } else if (boardName && typeof boardName === 'string') {
                // Jetson: Two-line format
                // Line 1: Board name (hostname)
                // Line 2: GPU name
                const gpuDisplayName = gpuName.includes('GPU') ? gpuName : `${gpuName} GPU`;
                systemInfoElem.innerHTML = `<b>${boardName}</b> (<code>${hostname}</code>)<br>${gpuDisplayName}`;
            } else if (productName && productName.includes('DGX')) {
                // DGX: Two-line format
                // Line 1: DGX Product name (hostname)
                // Line 2: GPU name
                const gpuDisplayName = gpuName.includes('GPU') ? gpuName : `${gpuName} GPU`;
                systemInfoElem.innerHTML = `<b>${productName}</b> (<code>${hostname}</code>)<br>${gpuDisplayName}`;
            } else if (productName && productName.includes('Mac')) {
                // Mac: Two-line format
                // Line 1: Product name CPU (hostname)
                // Line 2: GPU with cores
                const gpuText = gpuCores > 0 ? `${gpuName} ${gpuCores}-core GPU` : `${gpuName} GPU`;
                systemInfoElem.innerHTML = `${productName} ${cpuModel} (<code>${hostname}</code>)<br>${gpuText}`;
            } else if (productName && !productName.includes('Mac') && !productName.includes('DGX')) {
                // PC with product/motherboard name (branded or DIY)
                if (hasDiscreteGPU) {
                    // Has discrete GPU: Two-line format
                    // Line 1: Product Name (hostname) with CPU (no frequency)
                    // Line 2: GPU name
                    const gpuDisplayName = gpuName.includes('GPU') ? gpuName : `${gpuName} GPU`;
                    systemInfoElem.innerHTML = `<b>${productName}</b> (<code>${hostname}</code>) with ${cpuModelShort}<br>${gpuDisplayName}`;
                } else {
                    // No discrete GPU (integrated only): Two-line format
                    // Line 1: Product Name (hostname)
                    // Line 2: with CPU (including frequency)
                    systemInfoElem.innerHTML = `<b>${productName}</b> (<code>${hostname}</code>)<br>with ${cpuModel}`;
                }
            } else {
                // Generic PC
                if (hasDiscreteGPU) {
                    // Has discrete GPU: Two-line format
                    // Line 1: Hostname with CPU (no frequency)
                    // Line 2: GPU name
                    const gpuDisplayName = gpuName.includes('GPU') ? gpuName : `${gpuName} GPU`;
                    systemInfoElem.innerHTML = `<code>${hostname}</code> with ${cpuModelShort}<br>${gpuDisplayName}`;
                } else {
                    // No discrete GPU: Two-line format
                    // Line 1: Hostname
                    // Line 2: with CPU (including frequency)
                    systemInfoElem.innerHTML = `<code>${hostname}</code><br>with ${cpuModel}`;
                }
            }

            // Update product image (icon in header)
            const productImg = document.getElementById('systemProductImage');
            const imageSrc = getSystemProductImage(stats);
            if (imageSrc) {
                productImg.src = imageSrc;
                productImg.style.display = 'block';

                // Also set as background image for the card
                const systemStatsCard = document.getElementById('systemStatsCard');
                systemStatsCard.style.setProperty('--system-bg-image', `url('${imageSrc}')`);
            } else {
                productImg.style.display = 'none';
            }

            // CPU (Blue) - circular ring gauge
            const cpuPercent = stats.cpu_percent || 0;
            updateMetricText(document.getElementById('cpuUtil'), `${cpuPercent.toFixed(1)}%`);
            setRingProgress('cpuRing', cpuPercent, 263.9);

            // RAM (Orange) - circular ring gauge
            const ramUsed = stats.ram_used_gb || 0;
            const ramTotal = stats.ram_total_gb || 0;
            const ramPercent = stats.ram_percent || 0;
            updateMetricHTML(document.getElementById('ramUsage'), `${ramUsed.toFixed(1)}<span class="stat-value-denominator">/${ramTotal.toFixed(1)}GB</span>`);
            setRingProgress('ramRing', ramPercent, 263.9);

            // Thermal (CPU / iGPU / NPU) - current values
            const fmtTemp = (t) => (t === null || t === undefined) ? 'N/A' : `${t.toFixed(1)}\u00b0C`;
            updateMetricText(document.getElementById('cpuTempValue'), fmtTemp(stats.cpu_temp_c));
            updateMetricText(document.getElementById('gpuTempValue'), fmtTemp(stats.gpu_temp_c));
            updateMetricText(document.getElementById('npuTempValue'), fmtTemp(stats.npu_temp_c));

            // Sparklines
            if (stats.history) {
                const canvases = {
                    cpu: document.getElementById('cpuSparkline'),
                    ram: document.getElementById('ramSparkline'),
                    thermal: document.getElementById('thermalChart')
                };

                // Set canvas dimensions (use offsetWidth with fallback to parent width)
                Object.values(canvases).forEach(canvas => {
                    const width = canvas.offsetWidth || canvas.parentElement.offsetWidth || 300;
                    const height = canvas.offsetHeight || 40;

                    // Only update if dimensions changed (avoid unnecessary canvas clears)
                    if (canvas.width !== width || canvas.height !== height) {
                        canvas.width = width;
                        canvas.height = height;
                    }
                });

                drawSparkline(canvases.cpu, stats.history.cpu_util, themeColor('--thermal-cpu'));
                drawSparkline(canvases.ram, stats.history.ram_used, themeColor('--accent-color-2'));

                drawMultiSparkline(canvases.thermal, [
                    { data: stats.history.cpu_temp, color: themeColor('--thermal-cpu') },
                    { data: stats.history.gpu_temp, color: themeColor('--thermal-igpu') },
                    { data: stats.history.npu_temp, color: themeColor('--thermal-npu') },
                ]);
            }
        }

        // Set a circular ring gauge's fill amount (0-100%) via stroke-dashoffset
        function setRingProgress(elementId, percent, circumference) {
            const ring = document.getElementById(elementId);
            if (!ring) return;
            const clamped = Math.max(0, Math.min(100, percent || 0));
            const offset = circumference * (1 - clamped / 100);
            ring.style.strokeDashoffset = offset;
            ring.classList.toggle('high', clamped >= 85);
        }

        // Periodic check for pending GPU stats updates
        setInterval(() => {
            if (pendingGPUStats) {
                const now = Date.now();
                const intervalMs = settings.gpuUpdateInterval * 1000;

                if (now - lastGPUUpdate >= intervalMs) {
                    updateGPUStats(pendingGPUStats);
                    lastGPUUpdate = now;
                    pendingGPUStats = null;
                }
            }
        }, 50);  // Check every 50ms for smooth updates

        // WebSocket Connection
        function connectWebSocket() {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                return;
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws?session_id=${encodeURIComponent(sessionId)}`;

            websocket = new WebSocket(wsUrl);

            websocket.onopen = () => {
                console.log('WebSocket connected');
                updateStatus('Connected', 'connected');

                // Send current model/API settings to server after WebSocket connects
                // This ensures auto-selected models are properly initialized on the server side
                // Fix for race condition: page load might auto-select model before WS connects
                const currentModel = modelSelect.value;
                if (currentModel) {
                    console.log('Initializing server with current model:', currentModel);
                    applyApiSettings({ showFeedback: false });
                }
            };

            websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'vlm_response') {
                    // Trigger animations for new messages based on settings
                    if (data.text !== lastText) {
                        resultText.classList.remove('fade');
                        resultText.classList.remove('new-message');

                        // Apply pop-in animation if enabled
                        if (settings.popIn) {
                            // Force reflow to restart animation
                            void resultText.offsetWidth;

                            resultText.classList.add('new-message');

                            // Add glow effect if enabled
                            if (settings.glow) {
                                resultText.classList.add('with-glow');
                            }

                            // Remove animation classes after it completes
                            setTimeout(() => {
                                resultText.classList.remove('new-message');
                                resultText.classList.remove('with-glow');
                            }, 400);
                        }

                        if (fadeTimeout) {
                            clearTimeout(fadeTimeout);
                        }

                        // Start fade after 2 seconds if enabled
                        if (settings.fade) {
                            fadeTimeout = setTimeout(() => {
                                resultText.classList.add('fade');
                            }, 2000);
                        }
                    }

                    updateResultText(data.text);
                    lastText = data.text;

                    // Update video overlay (always, visibility is controlled by CSS)
                    videoOverlay.textContent = data.text;

                    if (data.metrics) {
                        metricsInline.style.display = 'flex';
                        latencyValue.textContent = Math.round(data.metrics.last_latency_ms);
                        avgLatencyValue.textContent = Math.round(data.metrics.avg_latency_ms);
                        countValue.textContent = data.metrics.total_inferences;
                    }
                    if (data.request_payload) {
                        const block = document.getElementById('requestPayloadDebug');
                        const pre = document.getElementById('requestPayloadContent');
                        if (block && pre && document.getElementById('debugShowRequestPayload')?.checked) {
                            block.style.display = 'block';
                            pre.textContent = JSON.stringify(data.request_payload, null, 2);
                        }
                    }
                    if (data.response_payload) {
                        const block = document.getElementById('responsePayloadDebug');
                        const pre = document.getElementById('responsePayloadContent');
                        if (block && pre && document.getElementById('debugShowResponsePayload')?.checked) {
                            block.style.display = 'block';
                            pre.textContent = JSON.stringify(data.response_payload, null, 2);
                        }
                    }
                } else if (data.type === 'gpu_stats') {
                    window.lastSystemStats = data.stats;  // Store for theme changes

                    // Throttle GPU stats updates based on configured interval
                    const now = Date.now();
                    const intervalMs = settings.gpuUpdateInterval * 1000;

                    if (now - lastGPUUpdate >= intervalMs) {
                        // Enough time has passed, update immediately
                        updateGPUStats(data.stats);
                        lastGPUUpdate = now;
                        pendingGPUStats = null;
                    } else {
                        // Too soon, store for later update
                        pendingGPUStats = data.stats;
                    }
                } else if (data.type === 'status') {
                    // Don't show status messages in result balloon (too flashy)
                    // Status is already shown in the header
                } else if (data.type === 'server_config') {
                    // Server sent its current configuration (model, api_base, prompt)
                    if (data.model) {
                        document.getElementById('modelName').textContent = data.model;
                        // Also update the model select if it matches
                        if (modelSelect.querySelector(`option[value="${data.model}"]`)) {
                            modelSelect.value = data.model;
                        }
                    }
                    if (data.api_base) {
                        apiBaseUrl.value = data.api_base;
                        checkApiKeyRequirement(data.api_base);
                        updateSystemStatsVisibility(data.api_base);
                    }
                    if (data.process_every != null && !isNaN(data.process_every)) {
                        processEvery.value = String(data.process_every);
                    }
                    const reqCb = document.getElementById('debugShowRequestPayload');
                    const resCb = document.getElementById('debugShowResponsePayload');
                    if ((reqCb?.checked || resCb?.checked) && websocket && websocket.readyState === WebSocket.OPEN) {
                        websocket.send(JSON.stringify({
                            type: 'set_debug',
                            show_request_payload: !!reqCb?.checked,
                            show_response_payload: !!resCb?.checked
                        }));
                    }
                } else if (data.type === 'model_updated') {
                    // Model was updated on server
                    if (data.model) {
                        document.getElementById('modelName').textContent = data.model;
                        console.log('Model updated to:', data.model);
                    }
                } else if (data.type === 'prompt_updated') {
                    // Prompt was updated on server (already handled in applyPrompt)
                    console.log('Prompt updated:', data.prompt);
                } else if (data.type === 'processing_updated') {
                    if (data.process_every != null) processEvery.value = String(data.process_every);
                    console.log('Processing interval updated:', data.process_every);
                }
            };

            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            websocket.onclose = () => {
                console.log('WebSocket disconnected, will reconnect...');
                websocket = null;
                updateStatus('Reconnecting...', 'disconnected');
                setTimeout(connectWebSocket, 2000);
            };
        }

        // Start - dispatch to webcam or RTSP based on active tab
        // Auto-reconnect: when the connection drops unexpectedly (network change,
        // interface flap, etc.), automatically retry instead of requiring the
        // user to manually click Start again. Only stops retrying if the user
        // explicitly clicks Stop (userStop() below sets userInitiatedStop=true).
        let userInitiatedStop = false;
        let reconnectTimer = null;
        let isAutoReconnecting = false;
        const RECONNECT_DELAY_MS = 3000;

        function attemptAutoReconnect(reconnectFn) {
            if (userInitiatedStop || reconnectTimer) return;
            updateStatus('Reconnecting...', 'processing');
            reconnectTimer = setTimeout(async () => {
                reconnectTimer = null;
                if (userInitiatedStop) return;
                isAutoReconnecting = true;
                try {
                    await stop();
                    await reconnectFn();
                } catch (e) {
                    console.log('Auto-reconnect attempt failed, will retry:', e);
                } finally {
                    isAutoReconnecting = false;
                }
                // If the attempt didn't end up connected (reconnectFn's own catch
                // block may have swallowed the error), schedule another attempt.
                if (!userInitiatedStop && (!peerConnection || peerConnection.iceConnectionState !== 'connected')) {
                    attemptAutoReconnect(reconnectFn);
                }
            }, RECONNECT_DELAY_MS);
        }

        // Called by the Stop button - distinguishes an intentional stop from
        // an unexpected disconnect, so auto-reconnect knows not to retry.
        function userStop() {
            userInitiatedStop = true;
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            stop();
            setSidebarDrawerCollapsed(false);
        }

        async function start() {
            userInitiatedStop = false;
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }

            // Auto-collapse the control drawer once settings are locked in and
            // streaming starts, so the video/stats get full screen real estate
            setSidebarDrawerCollapsed(true);

            // Cross-fade the big Start button out and the small Stop button in
            // (opacity-only: avoids fighting the existing translate(-50%,-50%)
            // centering transform, and is cleanly interruptible on rapid clicks).
            const bigBtn = document.getElementById('bigStartBtn');
            const smallStopBtn = document.getElementById('smallStopBtn');
            smallStopBtn.style.display = 'flex';
            smallStopBtn.classList.add('show');
            motionAnimate(smallStopBtn, { opacity: [0, 0.7] }, { duration: 0.25, easing: 'ease-out' });
            motionAnimate(bigBtn, { opacity: [1, 0] }, { duration: 0.2, easing: 'ease-in' }).then(() => {
                bigBtn.style.display = 'none';
            });

            // Live status border/glow on the video card itself
            document.getElementById('videoCard')?.classList.add('streaming-live');

            const activeTab = document.querySelector('.input-source-tab.active');
            const inputSource = activeTab ? activeTab.getAttribute('data-source') : 'webcam';

            if (inputSource === 'webcam') {
                await startWebcam();
            } else if (inputSource === 'rtsp') {
                await startRTSP();
            } else if (inputSource === 'videofile') {
                await startVideoFile();
            }
        }

        // Start WebRTC (Webcam mode)
        async function startWebcam() {
            try {
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }

                updateStatus('Requesting camera...', 'processing');

                // Use selected camera or default
                const videoConstraints = {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                };

                if (selectedCameraId) {
                    videoConstraints.deviceId = { exact: selectedCameraId };
                }

                localStream = await navigator.mediaDevices.getUserMedia({
                    video: videoConstraints,
                    audio: false
                });
                videoElement.srcObject = localStream;

                updateStatus('Connecting...', 'processing');

                // No STUN/TURN server needed - browser and server are on the same
                // machine (localhost), so host candidates alone are sufficient.
                // A public STUN server here would make the webcam feed silently
                // depend on internet access for no reason.
                peerConnection = new RTCPeerConnection({
                    iceServers: []
                });

                localStream.getTracks().forEach(track => {
                    peerConnection.addTrack(track, localStream);
                });

                peerConnection.ontrack = (event) => {
                    console.log('Received remote track');
                    if (event.track.kind === 'video') {
                        videoElement.srcObject = event.streams[0];
                        // Explicitly play to ensure continuous frame flow
                        videoElement.play().catch(err => {
                            console.error('Error playing video:', err);
                        });
                    }
                };

                peerConnection.oniceconnectionstatechange = () => {
                    console.log('ICE connection state:', peerConnection.iceConnectionState);
                    switch (peerConnection.iceConnectionState) {
                        case 'connected':
                            updateStatus('Streaming', 'connected');
                            break;
                        case 'disconnected':
                        case 'failed':
                        case 'closed':
                            updateStatus('Disconnected - reconnecting...', 'disconnected');
                            attemptAutoReconnect(startWebcam);
                            break;
                    }
                };

                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type,
                        session_id: sessionId,
                    }),
                });

                const answer = await response.json();
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

                isAnalysisRunning = true;
                updateStatus('Streaming', 'connected');

            } catch (error) {
                console.error('Error starting webcam:', error);
                updateStatus(`Error: ${error.message}`, 'disconnected');
                stop();
                if (!userInitiatedStop && !isAutoReconnecting) {
                    attemptAutoReconnect(startWebcam);
                }
            }
        }

        // Start RTSP mode
        async function startRTSP() {
            try {
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }

                const rtspUrl = document.getElementById('rtspUrl').value.trim();

                if (!rtspUrl) {
                    alert('Please enter an RTSP URL');
                    return;
                }

                updateStatus('Connecting to RTSP stream...', 'processing');

                // Create WebRTC peer connection (no local stream needed for RTSP)
                // No STUN/TURN needed here either - same-machine connection only.
                peerConnection = new RTCPeerConnection({
                    iceServers: []
                });

                // Receive processed video from server
                peerConnection.ontrack = (event) => {
                    console.log('Received RTSP stream from server');
                    if (event.track.kind === 'video') {
                        videoElement.srcObject = event.streams[0];
                        // Explicitly play the video to ensure continuous frame flow
                        videoElement.play().catch(err => {
                            console.error('Error playing video:', err);
                        });
                        updateStatus('Streaming', 'connected');
                    }
                };

                peerConnection.oniceconnectionstatechange = () => {
                    console.log('ICE connection state:', peerConnection.iceConnectionState);
                    switch (peerConnection.iceConnectionState) {
                        case 'connected':
                            updateStatus('Streaming', 'connected');
                            break;
                        case 'disconnected':
                        case 'failed':
                        case 'closed':
                            updateStatus('Disconnected - reconnecting...', 'disconnected');
                            attemptAutoReconnect(startRTSP);
                            break;
                    }
                };

                // Create offer (recvonly - we're only receiving video from server)
                peerConnection.addTransceiver('video', { direction: 'recvonly' });
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                // Wait for ICE gathering to complete before sending offer (with timeout)
                // This ensures all ICE candidates are embedded in the SDP
                console.log('Waiting for ICE gathering to complete...');
                await new Promise((resolve) => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        const checkState = () => {
                            if (peerConnection.iceGatheringState === 'complete') {
                                peerConnection.removeEventListener('icegatheringstatechange', checkState);
                                resolve();
                            }
                        };
                        peerConnection.addEventListener('icegatheringstatechange', checkState);

                        // Timeout after 5 seconds - proceed anyway to avoid hanging
                        setTimeout(() => {
                            peerConnection.removeEventListener('icegatheringstatechange', checkState);
                            console.warn('ICE gathering timeout - proceeding anyway');
                            resolve();
                        }, 5000);
                    }
                });
                console.log('ICE gathering state:', peerConnection.iceGatheringState, '- sending offer');

                // Send offer with RTSP URL and session to server
                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type,
                        rtsp_url: rtspUrl,
                        session_id: sessionId,
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Failed to connect to RTSP stream');
                }

                const answer = await response.json();
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

                isAnalysisRunning = true;

                // Disable RTSP URL field and Test button while streaming
                rtspUrlInput.disabled = true;
                document.getElementById('testRtspBtn').disabled = true;

            } catch (error) {
                console.error('Error starting RTSP:', error);
                updateStatus(`Error: ${error.message}`, 'disconnected');
                if (!isAutoReconnecting) {
                    alert('Failed to connect to RTSP stream: ' + error.message);
                }
                if (peerConnection) {
                    peerConnection.close();
                    peerConnection = null;
                }
                if (!userInitiatedStop && !isAutoReconnecting) {
                    attemptAutoReconnect(startRTSP);
                }
            }
        }

        // Start WebRTC (Video File mode) - plays a local video file instead
        // of a live camera, uploading it to the server once, then reusing
        // the saved path for reconnects/loops.
        let uploadedVideoFilePath = null;
        let uploadedVideoFileName = null;

        async function startVideoFile() {
            try {
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }

                const fileInput = document.getElementById('videoFileInput');
                const statusDiv = document.getElementById('videoFileStatus');
                const selectedFile = fileInput.files && fileInput.files[0];

                if (!selectedFile && !uploadedVideoFilePath) {
                    alert('Please choose a video file first');
                    return;
                }

                // Upload only if this is a newly-chosen file (not just a reconnect
                // using the same file we already uploaded)
                if (selectedFile && selectedFile.name !== uploadedVideoFileName) {
                    updateStatus('Uploading video file...', 'processing');
                    statusDiv.textContent = `Uploading ${selectedFile.name}...`;

                    const formData = new FormData();
                    formData.append('file', selectedFile);

                    const uploadResponse = await fetch('/api/video/upload', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!uploadResponse.ok) {
                        const err = await uploadResponse.json();
                        throw new Error(err.error || 'Failed to upload video file');
                    }

                    const uploadResult = await uploadResponse.json();
                    uploadedVideoFilePath = uploadResult.video_file_path;
                    uploadedVideoFileName = selectedFile.name;
                    statusDiv.textContent = `Ready: ${selectedFile.name}`;
                }

                updateStatus('Connecting to video file...', 'processing');

                // No STUN/TURN needed - same-machine connection only.
                peerConnection = new RTCPeerConnection({
                    iceServers: []
                });

                peerConnection.ontrack = (event) => {
                    console.log('Received video file stream from server');
                    if (event.track.kind === 'video') {
                        videoElement.srcObject = event.streams[0];
                        videoElement.play().catch(err => {
                            console.error('Error playing video:', err);
                        });
                        updateStatus('Streaming', 'connected');
                    }
                };

                peerConnection.oniceconnectionstatechange = () => {
                    console.log('ICE connection state:', peerConnection.iceConnectionState);
                    switch (peerConnection.iceConnectionState) {
                        case 'connected':
                            updateStatus('Streaming', 'connected');
                            break;
                        case 'disconnected':
                        case 'failed':
                        case 'closed':
                            updateStatus('Disconnected - reconnecting...', 'disconnected');
                            attemptAutoReconnect(startVideoFile);
                            break;
                    }
                };

                // recvonly - we're only receiving the processed video back from the server
                peerConnection.addTransceiver('video', { direction: 'recvonly' });
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                console.log('Waiting for ICE gathering to complete...');
                await new Promise((resolve) => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        const checkState = () => {
                            if (peerConnection.iceGatheringState === 'complete') {
                                peerConnection.removeEventListener('icegatheringstatechange', checkState);
                                resolve();
                            }
                        };
                        peerConnection.addEventListener('icegatheringstatechange', checkState);
                        setTimeout(() => {
                            peerConnection.removeEventListener('icegatheringstatechange', checkState);
                            console.warn('ICE gathering timeout - proceeding anyway');
                            resolve();
                        }, 5000);
                    }
                });
                console.log('ICE gathering state:', peerConnection.iceGatheringState, '- sending offer');

                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type,
                        video_file_path: uploadedVideoFilePath,
                        session_id: sessionId,
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Failed to play video file');
                }

                const answer = await response.json();
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

                isAnalysisRunning = true;

                fileInput.disabled = true;

            } catch (error) {
                console.error('Error starting video file:', error);
                updateStatus(`Error: ${error.message}`, 'disconnected');
                if (!isAutoReconnecting) {
                    alert('Failed to play video file: ' + error.message);
                }
                if (peerConnection) {
                    peerConnection.close();
                    peerConnection = null;
                }
                if (!userInitiatedStop && !isAutoReconnecting) {
                    attemptAutoReconnect(startVideoFile);
                }
            }
        }

        // Stop (handles both webcam and RTSP)
        async function stop() {
            // Reset overlay buttons - cross-fade back to the big Start button
            const bigBtn = document.getElementById('bigStartBtn');
            const smallStopBtn = document.getElementById('smallStopBtn');
            motionAnimate(smallStopBtn, { opacity: [0.7, 0] }, { duration: 0.15, easing: 'ease-in' }).then(() => {
                smallStopBtn.style.display = 'none';
                smallStopBtn.classList.remove('show');
            });
            bigBtn.style.display = 'flex';
            motionAnimate(bigBtn, { opacity: [0, 1] }, { duration: 0.25, easing: 'ease-out' });

            document.getElementById('videoCard')?.classList.remove('streaming-live');

            if (fadeTimeout) {
                clearTimeout(fadeTimeout);
                fadeTimeout = null;
            }

            // Stop webcam (if in webcam mode)
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }

            // Close WebRTC peer connection (used by both webcam and RTSP)
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }

            videoElement.srcObject = null;

            isAnalysisRunning = false;
            updateStatus('Connected', 'connected');

            // Re-enable RTSP URL field and Test button
            const rtspUrlInput = document.getElementById('rtspUrl');
            if (rtspUrlInput) {
                rtspUrlInput.disabled = false;
            }
            const testRtspBtn = document.getElementById('testRtspBtn');
            if (testRtspBtn) {
                testRtspBtn.disabled = false;
            }

            // Re-enable video file input
            const videoFileInput = document.getElementById('videoFileInput');
            if (videoFileInput) {
                videoFileInput.disabled = false;
            }

            const contentDiv = document.getElementById('resultTextContent');
            if (contentDiv) {
                contentDiv.textContent = '';
                contentDiv.innerHTML = '';
            } else {
                resultText.textContent = '';
                resultText.innerHTML = '';
            }
            resultText.classList.remove('fade');
            videoOverlay.textContent = '';
            lastText = '';
        }

        // Sidebar start/stop buttons removed - using overlay buttons instead
        // startBtn.addEventListener('click', start);
        // stopBtn.addEventListener('click', stop);

        // Load on page load
        window.addEventListener('load', async () => {
            // Detect services first, then fetch models
            await detectServices();
            fetchModels();
            connectWebSocket();

            // Initialize prompt display
            const initialPrompt = promptText.value.trim();
            if (initialPrompt) {
                const promptPreview = initialPrompt.length > 150 ? initialPrompt.substring(0, 150) + '...' : initialPrompt;
                currentPrompt.textContent = `Prompt: ${promptPreview}`;
            }

            // Initialize Lucide icons
            lucide.createIcons();
        });
