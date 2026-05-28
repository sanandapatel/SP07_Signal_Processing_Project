// ============================================================
// DSP ENGINE FOR SPECTRASCOPE — Professional Audio Analyzer
// ============================================================

// ── DSP CONFIGURATION ──────────────────────────────────────
let FS = 16000;                     // updated from AudioContext
const FFT_SIZE = 2048;
const SPECTROGRAM_FFT_SIZE = 1024;
const SPECTROGRAM_HOP = 512;
const BUFFER_DURATION = 3;          // seconds of ring buffer
let MAX_BUFFER_SIZE = FS * BUFFER_DURATION;
const NOISE_GATE_THRESHOLD = 0.003; // lowered for better sensitivity

const FFT_DISPLAY_MAX_HZ = 20000;  // cap FFT display at 20 kHz
const FFT_FLOOR_DB = -90;          // improved floor (was -120)
const OCTAVE_FLOOR_DB = -60;       // relative octave clamp
const SMOOTHING_ALPHA = 0.15;      // exponential smoothing: new = α*current + (1-α)*prev

const FREQ_RANGES = {
    'full': [0, 0],                 // set after FS known
    'narrowband': [300, 3400],
    'wideband': [50, 7000]
};

// ── DSP STATE ──────────────────────────────────────────────
let windowHanning = null;
let windowSum = 0;
let specWindow = null;
let specWindowSum = 0;
let freqBins = [];                  // frequency axis for FFT display

// Circular buffer for spectrogram
let audioRingBuffer = null;
let ringWritePos = 0;
let ringFilled = false;

let currentFreqRange = 'full';
let currentLowHz = 0;
let currentHighHz = 0;

// FFT processor instances
let fftProcessor = null;
let specFftProcessor = null;
let fftReal = [], fftImag = [], filteredReal = [], filteredImag = [];

// Smoothed FFT display buffer (exponential averaging)
let smoothedFFTdB = null;

// ── FFT PROCESSOR CLASS ────────────────────────────────────
// Radix-2 Cooley-Tukey FFT implementation
class FFT {
    constructor(size) {
        this.size = size;
        this.bitRev = new Uint32Array(size);
        this.cosTable = new Float64Array(size / 2);
        this.sinTable = new Float64Array(size / 2);
        this._initTables();
    }

    _initTables() {
        const levels = Math.round(Math.log2(this.size));
        for (let i = 0; i < this.size; i++) {
            this.bitRev[i] = this._reverseBits(i, levels);
        }
        for (let i = 0; i < this.size / 2; i++) {
            const angle = -2 * Math.PI * i / this.size;
            this.cosTable[i] = Math.cos(angle);
            this.sinTable[i] = Math.sin(angle);
        }
    }

    _reverseBits(x, bits) {
        let rev = 0;
        for (let i = 0; i < bits; i++) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        return rev;
    }

    transform(real, imag, inverse = false) {
        const n = this.size;
        const halfN = n / 2;

        // Bit-reversal permutation
        for (let i = 0; i < n; i++) {
            const j = this.bitRev[i];
            if (i < j) {
                let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
                tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
            }
        }

        // Butterfly stages
        for (let len = 2; len <= n; len <<= 1) {
            const half = len >> 1;
            const step = n / len;
            for (let i = 0; i < n; i += len) {
                for (let k = 0; k < half; k++) {
                    const twiddleIdx = k * step;
                    // For inverse FFT, conjugate the twiddle factor
                    const c = this.cosTable[twiddleIdx];
                    const s = inverse ? -this.sinTable[twiddleIdx] : this.sinTable[twiddleIdx];
                    const j = i + k + half;
                    const tre = c * real[j] - s * imag[j];
                    const tim = c * imag[j] + s * real[j];
                    real[j] = real[i + k] - tre;
                    imag[j] = imag[i + k] - tim;
                    real[i + k] += tre;
                    imag[i + k] += tim;
                }
            }
        }

        if (inverse) {
            const invN = 1.0 / n;
            for (let i = 0; i < n; i++) {
                real[i] *= invN;
                imag[i] *= invN;
            }
        }
    }

    realFFT(signal, outReal, outImag) {
        const n = this.size;
        for (let i = 0; i < n; i++) {
            outReal[i] = signal[i];
            outImag[i] = 0;
        }
        this.transform(outReal, outImag, false);
        // Zero negative frequencies (one-sided spectrum)
        for (let i = n / 2 + 1; i < n; i++) {
            outReal[i] = 0;
            outImag[i] = 0;
        }
    }

    realIFFT(inReal, inImag, outSignal) {
        const n = this.size;
        const fullReal = new Float64Array(n);
        const fullImag = new Float64Array(n);
        // Copy positive frequencies
        for (let i = 0; i <= n / 2; i++) {
            fullReal[i] = inReal[i];
            fullImag[i] = inImag[i];
        }
        // Reconstruct conjugate-symmetric negative frequencies
        for (let i = n / 2 + 1; i < n; i++) {
            fullReal[i] = inReal[n - i];
            fullImag[i] = -inImag[n - i];
        }
        this.transform(fullReal, fullImag, true);
        for (let i = 0; i < n; i++) {
            outSignal[i] = fullReal[i];
        }
    }
}


// ════════════════════════════════════════════════════════════
// INITIALIZATION
// ════════════════════════════════════════════════════════════
function initDSPWithSampleRate(newFs) {
    FS = newFs;
    MAX_BUFFER_SIZE = FS * BUFFER_DURATION;
    audioRingBuffer = new Float32Array(MAX_BUFFER_SIZE);
    ringWritePos = 0;
    ringFilled = false;

    // ── Hanning window for main FFT ──
    windowHanning = new Float32Array(FFT_SIZE);
    windowSum = 0;
    for (let i = 0; i < FFT_SIZE; i++) {
        windowHanning[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (FFT_SIZE - 1)));
        windowSum += windowHanning[i];
    }

    // ── Hanning window for spectrogram FFT ──
    specWindow = new Float32Array(SPECTROGRAM_FFT_SIZE);
    specWindowSum = 0;
    for (let i = 0; i < SPECTROGRAM_FFT_SIZE; i++) {
        specWindow[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (SPECTROGRAM_FFT_SIZE - 1)));
        specWindowSum += specWindow[i];
    }

    // ── Frequency bin centers ──
    freqBins = new Array(FFT_SIZE / 2 + 1);
    for (let i = 0; i <= FFT_SIZE / 2; i++) {
        freqBins[i] = i * FS / FFT_SIZE;
    }

    // ── Update frequency range presets ──
    FREQ_RANGES.full = [0, Math.min(FS / 2, FFT_DISPLAY_MAX_HZ)];
    currentLowHz = FREQ_RANGES[currentFreqRange][0];
    currentHighHz = FREQ_RANGES[currentFreqRange][1];

    // ── FFT processors ──
    fftProcessor = new FFT(FFT_SIZE);
    specFftProcessor = new FFT(SPECTROGRAM_FFT_SIZE);
    fftReal = new Float64Array(FFT_SIZE);
    fftImag = new Float64Array(FFT_SIZE);
    filteredReal = new Float64Array(FFT_SIZE);
    filteredImag = new Float64Array(FFT_SIZE);

    // ── Reset smoothed FFT buffer ──
    smoothedFFTdB = null;

    // ── Update UI elements if they exist ──
    if (typeof fftChart !== 'undefined' && fftChart) {
        fftChart.options.scales.x.title.text = `Frequency (Hz) — Fs = ${FS} Hz`;
        fftChart.update();
    }
    if (typeof modeBadge !== 'undefined' && modeBadge) {
        modeBadge.innerHTML = `<i class="fas fa-microphone-alt mr-1"></i> Live @ ${FS} Hz`;
    }
}


// ════════════════════════════════════════════════════════════
// DSP HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════

// ── RMS energy ──
function computeRMS(signal) {
    let sum = 0;
    for (let i = 0; i < signal.length; i++) {
        sum += signal[i] * signal[i];
    }
    return Math.sqrt(sum / signal.length);
}

// ── Frequency-domain masking ──
function applyFreqMask(real, imag, lowHz, highHz, fs, n_fft) {
    const n = real.length;
    for (let i = 0; i < n; i++) {
        const freq = i * fs / n_fft;
        if (freq < lowHz || freq > highHz) {
            real[i] = 0;
            imag[i] = 0;
        }
    }
}

// ── Linear magnitude → dB with configurable floor ──
function toDb(linearVals, floor = FFT_FLOOR_DB) {
    const db = new Array(linearVals.length);
    for (let i = 0; i < linearVals.length; i++) {
        const val = Math.max(linearVals[i], 1e-12);
        db[i] = Math.max(20.0 * Math.log10(val), floor);
    }
    return db;
}


// ── 1/1 Octave band analysis with RELATIVE normalization ──
function octaveBands(fftMagnitudeLinear, freqsArray) {
    const bandsCenter = [125, 250, 500, 1000, 2000, 4000];
    const rawEnergies = [];
    const sqrt2 = Math.SQRT2;

    // Accumulate power in each octave band
    for (const fc of bandsCenter) {
        const fLow = fc / sqrt2;
        const fHigh = fc * sqrt2;
        let power = 0;
        let binCount = 0;
        for (let i = 0; i < freqsArray.length; i++) {
            if (freqsArray[i] >= fLow && freqsArray[i] <= fHigh) {
                power += fftMagnitudeLinear[i] * fftMagnitudeLinear[i];
                binCount++;
            }
        }
        // Normalize by bin count to get average power (prevents bias from wider bands)
        const avgPower = binCount > 0 ? power / binCount : 1e-12;
        const dbVal = 10.0 * Math.log10(Math.max(avgPower, 1e-12));
        rawEnergies.push(dbVal);
    }

    // ── RELATIVE NORMALIZATION ──
    // Find max energy, normalize all bands relative to it
    const maxEnergy = Math.max(...rawEnergies);
    const normalizedEnergies = rawEnergies.map(e => {
        const rel = e - maxEnergy;
        return Math.max(rel, OCTAVE_FLOOR_DB); // clamp at -60 dB relative
    });

    return { bands: bandsCenter, energies: normalizedEnergies };
}


// ── Frequency-domain bandpass filter via FFT→mask→IFFT ──
function filterSignalFreqDomain(signal, lowHz, highHz, fs, n_fft) {
    const real = new Float64Array(n_fft);
    const imag = new Float64Array(n_fft);
    for (let i = 0; i < n_fft; i++) {
        real[i] = signal[i];
        imag[i] = 0;
    }

    fftProcessor.transform(real, imag, false);

    // Apply bandpass mask to positive frequencies
    for (let i = 0; i <= n_fft / 2; i++) {
        const freq = i * fs / n_fft;
        if (freq < lowHz || freq > highHz) {
            real[i] = 0;
            imag[i] = 0;
        }
    }
    // Reconstruct conjugate-symmetric negative frequencies
    for (let i = n_fft / 2 + 1; i < n_fft; i++) {
        real[i] = real[n_fft - i];
        imag[i] = -imag[n_fft - i];
    }

    fftProcessor.transform(real, imag, true);

    const out = new Float64Array(n_fft);
    for (let i = 0; i < n_fft; i++) {
        out[i] = real[i];
    }
    return out;
}


// ── Windowed FFT → linear magnitude spectrum ──
function computeFFTLinear(signal, windowArr, windowSumVal, fftSize) {
    const real = new Float64Array(fftSize);
    const imag = new Float64Array(fftSize);

    // Apply window
    for (let i = 0; i < fftSize; i++) {
        const idx = i < signal.length ? i : signal.length - 1;
        real[i] = signal[idx] * windowArr[i];
        imag[i] = 0;
    }

    fftProcessor.transform(real, imag, false);

    // Compute one-sided magnitude spectrum, properly scaled
    const nBins = fftSize / 2 + 1;
    const mag = new Float64Array(nBins);
    for (let i = 0; i < nBins; i++) {
        const absVal = Math.hypot(real[i], imag[i]);
        // Scale: multiply by 2 for one-sided spectrum, divide by window sum
        let scaled = absVal * 2.0 / windowSumVal;
        // DC bin should not be doubled
        if (i === 0) scaled *= 0.5;
        // Nyquist bin should not be doubled
        if (i === fftSize / 2) scaled *= 0.5;
        mag[i] = scaled;
    }
    return mag;
}


// ── Exponential smoothing for FFT display ──
function smoothFFT(currentDb) {
    if (!smoothedFFTdB || smoothedFFTdB.length !== currentDb.length) {
        // Initialize smoothed buffer from first frame
        smoothedFFTdB = currentDb.slice();
        return smoothedFFTdB;
    }
    for (let i = 0; i < currentDb.length; i++) {
        smoothedFFTdB[i] = SMOOTHING_ALPHA * currentDb[i] + (1.0 - SMOOTHING_ALPHA) * smoothedFFTdB[i];
    }
    return smoothedFFTdB;
}


// ── Generate clean engineering frequency axis ticks ──
// Returns { labels, tickIndices } for Chart.js
function generateFreqAxisTicks(maxFreqHz, numBins, fsHz, fftSize) {
    // Determine good tick spacing based on max display frequency
    let tickSpacing;
    if (maxFreqHz <= 1000) {
        tickSpacing = 100;
    } else if (maxFreqHz <= 5000) {
        tickSpacing = 500;
    } else if (maxFreqHz <= 10000) {
        tickSpacing = 1000;
    } else {
        tickSpacing = 2000;
    }

    const ticks = [];
    for (let f = 0; f <= maxFreqHz; f += tickSpacing) {
        ticks.push(f);
    }
    // Always include the max if not already there
    if (ticks[ticks.length - 1] < maxFreqHz) {
        ticks.push(maxFreqHz);
    }

    return { ticks, tickSpacing };
}

// Returns the display-limited bin count (up to 20 kHz)
function getDisplayBinCount() {
    const maxHz = Math.min(FS / 2, FFT_DISPLAY_MAX_HZ);
    return Math.min(FFT_SIZE / 2 + 1, Math.floor(maxHz * FFT_SIZE / FS) + 1);
}


// ════════════════════════════════════════════════════════════
// RING BUFFER MANAGEMENT
// ════════════════════════════════════════════════════════════

function pushToAudioRingBuffer(samples) {
    if (!audioRingBuffer) return;
    for (let i = 0; i < samples.length; i++) {
        audioRingBuffer[ringWritePos] = samples[i];
        ringWritePos = (ringWritePos + 1) % MAX_BUFFER_SIZE;
        if (ringWritePos === 0) ringFilled = true;
    }
}

function getAudioRingBufferCopy() {
    if (!audioRingBuffer) return null;
    const totalSamples = ringFilled ? MAX_BUFFER_SIZE : ringWritePos;
    if (totalSamples === 0) return null;
    const bufferCopy = new Float32Array(totalSamples);
    if (ringFilled) {
        const firstPart = MAX_BUFFER_SIZE - ringWritePos;
        bufferCopy.set(audioRingBuffer.subarray(ringWritePos, ringWritePos + firstPart), 0);
        bufferCopy.set(audioRingBuffer.subarray(0, ringWritePos), firstPart);
    } else {
        bufferCopy.set(audioRingBuffer.subarray(0, ringWritePos), 0);
    }
    return { bufferCopy, totalSamples };
}


// ════════════════════════════════════════════════════════════
// SPECTROGRAM COMPUTATION (STFT)
// ════════════════════════════════════════════════════════════

function computeSpectrogramFromBuffer(bufferSamples, lengthSamples, fs, fftSize, hop) {
    if (lengthSamples < fftSize) return { spectrogram: [], freqs: [], times: [] };

    const nFrames = 1 + Math.floor((lengthSamples - fftSize) / hop);
    const nBins = fftSize / 2 + 1;

    const freqs = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) {
        freqs[i] = i * fs / fftSize;
    }

    const times = new Float32Array(nFrames);
    const spectrogram = [];

    // Reusable buffers for performance
    const frameReal = new Float64Array(fftSize);
    const frameImag = new Float64Array(fftSize);

    for (let frame = 0; frame < nFrames; frame++) {
        const start = frame * hop;
        times[frame] = start / fs;

        // Window the frame
        for (let i = 0; i < fftSize; i++) {
            frameReal[i] = (bufferSamples[start + i] || 0) * specWindow[i];
            frameImag[i] = 0;
        }

        specFftProcessor.transform(frameReal, frameImag, false);

        // Compute magnitude in dB
        const frameMag = new Float32Array(nBins);
        for (let i = 0; i < nBins; i++) {
            let mag = Math.hypot(frameReal[i], frameImag[i]) * 2.0 / specWindowSum;
            if (i === 0) mag *= 0.5;
            if (i === fftSize / 2) mag *= 0.5;
            const db = 20.0 * Math.log10(Math.max(mag, 1e-12));
            frameMag[i] = Math.max(db, FFT_FLOOR_DB);
        }
        spectrogram.push(frameMag);
    }

    return { spectrogram, freqs, times };
}
