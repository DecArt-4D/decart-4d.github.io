/**
 * Playback synchronization module for viser viewer
 * Syncs playback time when switching between decay levels
 */

const PlaybackSync = (function() {
    let pendingPlaybackTime = null;
    let syncInterval = null;
    const MAX_SYNC_ATTEMPTS = 100;
    const SYNC_INTERVAL_MS = 50;

    /**
     * Find the playback slider (range input) in the iframe
     */
    function findPlaybackSlider(iframeDoc) {
        if (!iframeDoc) return null;

        // Try to find by Mantine slider class first
        const mantineSlider = iframeDoc.querySelector('.mantine-Slider-thumb');
        if (mantineSlider) {
            const container = mantineSlider.closest('.mantine-Slider-root');
            if (container) {
                // Mantine uses a hidden input or we need to interact with the track
                const track = container.querySelector('.mantine-Slider-track');
                if (track) return { type: 'mantine', element: container, track };
            }
        }

        // Try standard range inputs
        const rangeInputs = Array.from(iframeDoc.querySelectorAll('input[type="range"]'));
        for (const input of rangeInputs) {
            const max = parseFloat(input.max || '0');
            if (max > 0) {
                return { type: 'range', element: input };
            }
        }

        return null;
    }

    /**
     * Read current playback time from iframe
     */
    function readPlaybackTime(iframe) {
        try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
            if (!iframeDoc) return null;

            const slider = findPlaybackSlider(iframeDoc);
            if (!slider) return null;

            if (slider.type === 'range') {
                const value = parseFloat(slider.element.value);
                return Number.isFinite(value) ? value : null;
            }

            if (slider.type === 'mantine') {
                // Try to read from aria-valuenow or style
                const thumb = slider.element.querySelector('.mantine-Slider-thumb');
                if (thumb) {
                    const ariaValue = thumb.getAttribute('aria-valuenow');
                    if (ariaValue) {
                        const value = parseFloat(ariaValue);
                        return Number.isFinite(value) ? value : null;
                    }
                }

                // Try to calculate from thumb position
                const bar = slider.element.querySelector('.mantine-Slider-bar');
                if (bar) {
                    const width = parseFloat(bar.style.width);
                    if (Number.isFinite(width)) {
                        // Assuming max is around 80 frames at 30fps = ~2.67s
                        // This is approximate
                        return (width / 100) * 2.67;
                    }
                }
            }
        } catch (err) {
            console.warn('[PlaybackSync] Error reading playback time:', err);
        }
        return null;
    }

    /**
     * Set playback time in iframe
     */
    function setPlaybackTime(iframe, targetTime) {
        try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
            if (!iframeDoc) return false;

            const slider = findPlaybackSlider(iframeDoc);
            if (!slider) return false;

            if (slider.type === 'range') {
                const min = parseFloat(slider.element.min || '0');
                const max = parseFloat(slider.element.max || `${targetTime}`);
                const clamped = Math.min(Math.max(targetTime, min), max);

                slider.element.value = String(clamped);
                slider.element.dispatchEvent(new Event('input', { bubbles: true }));
                slider.element.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            }

            if (slider.type === 'mantine') {
                // For Mantine sliders, we need to simulate mouse events on the track
                const track = slider.track || slider.element.querySelector('.mantine-Slider-track');
                if (track) {
                    const rect = track.getBoundingClientRect();
                    const maxTime = 2.67; // Approximate max time
                    const ratio = Math.min(Math.max(targetTime / maxTime, 0), 1);
                    const clientX = rect.left + rect.width * ratio;
                    const clientY = rect.top + rect.height / 2;

                    // Simulate click at the target position
                    const mouseDown = new MouseEvent('mousedown', {
                        bubbles: true,
                        clientX,
                        clientY,
                    });
                    const mouseUp = new MouseEvent('mouseup', {
                        bubbles: true,
                        clientX,
                        clientY,
                    });

                    track.dispatchEvent(mouseDown);
                    track.dispatchEvent(mouseUp);
                    return true;
                }
            }
        } catch (err) {
            console.warn('[PlaybackSync] Error setting playback time:', err);
        }
        return false;
    }

    /**
     * Try to click play button if paused
     */
    function ensurePlaying(iframe) {
        try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
            if (!iframeDoc) return;

            // Strategy 1: Look for play icon by class
            const playIcon = iframeDoc.querySelector('[class*="player-play"]') ||
                            iframeDoc.querySelector('svg[class*="player-play"]');

            if (playIcon) {
                const button = playIcon.closest('button');
                if (button) {
                    console.log('[PlaybackSync] Clicking play button');
                    button.click();
                    return;
                }
            }

            // Strategy 2: Find all SVGs and check their class
            const svgs = iframeDoc.querySelectorAll('svg');
            for (const svg of svgs) {
                const className = svg.getAttribute('class') || '';
                if (className.includes('player-play')) {
                    const button = svg.closest('button');
                    if (button) {
                        console.log('[PlaybackSync] Clicking play button (via SVG class)');
                        button.click();
                        return;
                    }
                }
            }

            // Strategy 3: Look for ActionIcon buttons near the slider
            const slider = iframeDoc.querySelector('.mantine-Slider-root');
            if (slider) {
                const parent = slider.parentElement;
                if (parent) {
                    const buttons = parent.querySelectorAll('button');
                    for (const btn of buttons) {
                        // Look for a button with an SVG that could be play
                        const svg = btn.querySelector('svg');
                        if (svg) {
                            console.log('[PlaybackSync] Found button near slider, clicking');
                            btn.click();
                            return;
                        }
                    }
                }
            }

            // Strategy 4: Try spacebar keypress to toggle play/pause
            console.log('[PlaybackSync] Trying spacebar to toggle play');
            const keyEvent = new KeyboardEvent('keydown', {
                key: ' ',
                code: 'Space',
                keyCode: 32,
                which: 32,
                bubbles: true
            });
            iframeDoc.body.dispatchEvent(keyEvent);

            console.log('[PlaybackSync] Play button not found, tried spacebar');
        } catch (err) {
            console.warn('[PlaybackSync] Error ensuring playback:', err);
        }
    }

    /**
     * Save current playback time before switching
     */
    function savePlaybackTime(iframe) {
        const time = readPlaybackTime(iframe);
        if (Number.isFinite(time)) {
            pendingPlaybackTime = time;
            console.log('[PlaybackSync] Saved playback time:', time);
        }
        return time;
    }

    /**
     * Restore playback time after iframe loads
     */
    function restorePlaybackTime(iframe) {
        if (!Number.isFinite(pendingPlaybackTime)) {
            console.log('[PlaybackSync] No pending playback time to restore');
            return;
        }

        const targetTime = pendingPlaybackTime;
        let attempts = 0;

        // Clear any existing sync interval
        if (syncInterval) {
            clearInterval(syncInterval);
        }

        console.log('[PlaybackSync] Starting sync to time:', targetTime);

        syncInterval = setInterval(() => {
            attempts++;

            // Try to set the playback time
            const success = setPlaybackTime(iframe, targetTime);

            if (success) {
                // Verify the time was set
                const currentTime = readPlaybackTime(iframe);
                if (currentTime !== null && Math.abs(currentTime - targetTime) < 0.1) {
                    console.log('[PlaybackSync] Successfully synced to:', currentTime);
                    pendingPlaybackTime = null;
                    clearInterval(syncInterval);
                    syncInterval = null;
                    // Small delay before clicking play to let UI settle
                    setTimeout(() => ensurePlaying(iframe), 100);
                    return;
                }
            }

            if (attempts >= MAX_SYNC_ATTEMPTS) {
                console.warn('[PlaybackSync] Max attempts reached, giving up');
                pendingPlaybackTime = null;
                clearInterval(syncInterval);
                syncInterval = null;
            }
        }, SYNC_INTERVAL_MS);
    }

    /**
     * Cancel any pending sync operation
     */
    function cancelSync() {
        if (syncInterval) {
            clearInterval(syncInterval);
            syncInterval = null;
        }
        pendingPlaybackTime = null;
    }

    /**
     * Get current pending time
     */
    function getPendingTime() {
        return pendingPlaybackTime;
    }

    // Public API
    return {
        savePlaybackTime,
        restorePlaybackTime,
        readPlaybackTime,
        cancelSync,
        getPendingTime,
    };
})();

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PlaybackSync;
}
