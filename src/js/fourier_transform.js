// ============================================================
// fourier_transform.js — Fourier Transform Visualizer
// ============================================================

// --- GLOBAL VARIABLES ---
let timeDomainChart, magnitudeChart, phaseChart;

// Wave colors matching the HTML
const WAVE_COLORS = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00'];

// Sampling parameters
const SAMPLING_RATE = 256; // Hz - Fixed sampling rate for display
const MAX_SAMPLES_FOR_DFT = 512; // Maximum samples for DFT to maintain performance

// --- DFT IMPLEMENTATION ---

/**
 * Cooley-Tukey FFT algorithm for power-of-2 sizes
 * Much faster than DFT for large N
 * 
 * @param {number[]} signal - Real-valued input signal
 * @returns {Object} - Object with real (re) and imaginary (im) parts arrays
 */
function fft(signal) {
    const N = signal.length;
    
    // Check if N is a power of 2
    if (N & (N - 1)) {
        // Not a power of 2, fall back to DFT
        return dft(signal);
    }
    
    // Bit-reverse copy
    const re = new Array(N);
    const im = new Array(N).fill(0);
    
    for (let i = 0; i < N; i++) {
        let j = 0;
        let bit = N >> 1;
        let ii = i;
        while (bit > 0) {
            if (ii & 1) j |= bit;
            ii >>= 1;
            bit >>= 1;
        }
        re[j] = signal[i];
    }
    
    // Butterfly operations
    for (let step = 2; step <= N; step <<= 1) {
        const halfStep = step >> 1;
        const angleInc = -2 * Math.PI / step;
        
        for (let group = 0; group < N; group += step) {
            for (let i = 0; i < halfStep; i++) {
                const angle = angleInc * i;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                
                const evenRe = re[group + i];
                const evenIm = im[group + i];
                const oddRe = re[group + i + halfStep] * cos - im[group + i + halfStep] * sin;
                const oddIm = re[group + i + halfStep] * sin + im[group + i + halfStep] * cos;
                
                re[group + i] = evenRe + oddRe;
                im[group + i] = evenIm + oddIm;
                re[group + i + halfStep] = evenRe - oddRe;
                im[group + i + halfStep] = evenIm - oddIm;
            }
        }
    }
    
    return { re, im };
}

/**
 * Discrete Fourier Transform (DFT) - O(N²) implementation
 * Input: real-valued time domain signal
 * Output: complex frequency domain values
 * 
 * @param {number[]} signal - Real-valued input signal
 * @returns {Object} - Object with real (re) and imaginary (im) parts arrays
 */
function dft(signal) {
    const N = signal.length;
    const re = new Array(N).fill(0);
    const im = new Array(N).fill(0);
    
    for (let k = 0; k < N; k++) {
        let sumRe = 0;
        let sumIm = 0;
        for (let n = 0; n < N; n++) {
            const angle = -2 * Math.PI * k * n / N;
            sumRe += signal[n] * Math.cos(angle);
            sumIm += signal[n] * Math.sin(angle);
        }
        re[k] = sumRe;
        im[k] = sumIm;
    }
    
    return { re, im };
}

/**
 * Compute magnitude spectrum from complex DFT output
 * Returns single-sided spectrum (positive frequencies only)
 * 
 * @param {Object} dftResult - Object with re and im arrays
 * @returns {number[]} - Magnitude values for positive frequencies
 */
function computeMagnitudeSpectrum(dftResult) {
    const N = dftResult.re.length;
    const magnitudes = [];
    
    // Only return positive frequencies (0 to N/2)
    for (let k = 0; k <= N / 2; k++) {
        const mag = Math.sqrt(dftResult.re[k] ** 2 + dftResult.im[k] ** 2);
        // Normalize by N/2 for single-sided spectrum
        magnitudes.push(2 * mag / N);
    }
    
    // DC component should not be doubled
    magnitudes[0] = magnitudes[0] / 2;
    if (N % 2 === 0) {
        magnitudes[magnitudes.length - 1] = magnitudes[magnitudes.length - 1] / 2;
    }
    
    return magnitudes;
}

/**
 * Compute phase spectrum from complex DFT output
 * Returns single-sided spectrum (positive frequencies only)
 * 
 * @param {Object} dftResult - Object with re and im arrays
 * @returns {number[]} - Phase values in degrees for positive frequencies
 */
function computePhaseSpectrum(dftResult) {
    const N = dftResult.re.length;
    const phases = [];
    
    for (let k = 0; k <= N / 2; k++) {
        const phaseRad = Math.atan2(dftResult.im[k], dftResult.re[k]);
        const phaseDeg = phaseRad * 180 / Math.PI;
        phases.push(phaseDeg);
    }
    
    return phases;
}

// --- SIGNAL GENERATION ---

/**
 * Generate a sine wave
 * 
 * @param {number} frequency - Frequency in Hz
 * @param {number} amplitude - Peak amplitude
 * @param {number} phaseDeg - Phase in degrees
 * @param {number} samplingRate - Sampling rate in Hz
 * @param {number} numSamples - Number of samples
 * @returns {number[]} - Generated signal
 */
function generateSineWave(frequency, amplitude, phaseDeg, samplingRate, numSamples) {
    const signal = [];
    const phaseRad = phaseDeg * Math.PI / 180;
    
    for (let n = 0; n < numSamples; n++) {
        const t = n / samplingRate;
        const value = amplitude * Math.sin(2 * Math.PI * frequency * t + phaseRad);
        signal.push(value);
    }
    
    return signal;
}

/**
 * Generate the composite signal from all enabled waves
 * 
 * @returns {Object} - Object with time array, signal array, and individual waves
 */
function generateCompositeSignal() {
    const numSamples = parseInt(document.getElementById('numSamples').value);
    const addNoise = document.getElementById('addNoise').checked;
    const noiseLevel = parseFloat(document.getElementById('noiseLevel').value);
    
    // Initialize signal and time arrays
    const time = [];
    const signal = new Array(numSamples).fill(0);
    const individualWaves = [];
    
    // Generate time array
    for (let n = 0; n < numSamples; n++) {
        time.push(n / SAMPLING_RATE);
    }
    
    // Add each enabled wave
    for (let i = 1; i <= 4; i++) {
        const enabled = document.getElementById(`enableWave${i}`).checked;
        if (!enabled) continue;
        
        const frequency = parseInt(document.getElementById(`freq${i}`).value);
        const amplitude = parseFloat(document.getElementById(`amp${i}`).value);
        const phase = parseInt(document.getElementById(`phase${i}`).value);
        
        if (amplitude > 0) {
            const wave = generateSineWave(frequency, amplitude, phase, SAMPLING_RATE, numSamples);
            individualWaves.push({
                frequency,
                amplitude,
                phase,
                color: WAVE_COLORS[i - 1],
                data: wave
            });
            
            // Add to composite signal
            for (let n = 0; n < numSamples; n++) {
                signal[n] += wave[n];
            }
        }
    }
    
    // Add noise if enabled
    if (addNoise) {
        for (let n = 0; n < numSamples; n++) {
            signal[n] += randomGaussian(0, noiseLevel);
        }
    }
    
    return { time, signal, individualWaves, numSamples };
}

// --- CHART INITIALIZATION ---

function initCharts() {
    // Time Domain Chart
    const timeCtx = document.getElementById('timeDomainChart').getContext('2d');
    timeDomainChart = new Chart(timeCtx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Composite Signal',
                    data: [],
                    borderColor: '#1976d2',
                    backgroundColor: 'rgba(25, 118, 210, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'Wave 1',
                    data: [],
                    borderColor: WAVE_COLORS[0],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    hidden: false
                },
                {
                    label: 'Wave 2',
                    data: [],
                    borderColor: WAVE_COLORS[1],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    hidden: false
                },
                {
                    label: 'Wave 3',
                    data: [],
                    borderColor: WAVE_COLORS[2],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    hidden: true
                },
                {
                    label: 'Wave 4',
                    data: [],
                    borderColor: WAVE_COLORS[3],
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    hidden: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Time (s)' }
                },
                y: {
                    title: { display: true, text: 'Amplitude' },
                    min: -25,
                    max: 25
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        filter: function(item, data) {
                            // Only show legend for visible waves
                            const dataset = data.datasets[item.datasetIndex];
                            return !dataset.hidden;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return `t = ${context[0].parsed.x.toFixed(4)} s`;
                        }
                    }
                }
            }
        }
    });

    // Magnitude Spectrum Chart
    const magCtx = document.getElementById('magnitudeChart').getContext('2d');
    magnitudeChart = new Chart(magCtx, {
        type: 'bar',
        data: {
            datasets: [{
                label: 'Magnitude',
                data: [],
                backgroundColor: 'rgba(25, 118, 210, 0.7)',
                borderColor: '#1976d2',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Frequency (Hz)' },
                    min: 0
                },
                y: {
                    title: { display: true, text: '|X(f)|' },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return `f = ${context[0].parsed.x.toFixed(1)} Hz`;
                        },
                        label: function(context) {
                            return `Magnitude: ${context.parsed.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });

    // Phase Spectrum Chart
    const phaseCtx = document.getElementById('phaseChart').getContext('2d');
    phaseChart = new Chart(phaseCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Phase',
                data: [],
                backgroundColor: 'rgba(46, 125, 50, 0.7)',
                borderColor: '#2e7d32',
                borderWidth: 1,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Frequency (Hz)' },
                    min: 0
                },
                y: {
                    title: { display: true, text: 'Phase (degrees)' },
                    min: -180,
                    max: 180,
                    ticks: {
                        stepSize: 45
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return `f = ${context[0].parsed.x.toFixed(1)} Hz`;
                        },
                        label: function(context) {
                            return `Phase: ${context.parsed.y.toFixed(1)}°`;
                        }
                    }
                }
            }
        }
    });
}

// --- UPDATE FUNCTIONS ---

function updateVisualizations() {
    const { time, signal, individualWaves, numSamples } = generateCompositeSignal();
    
    // Update Time Domain Chart
    timeDomainChart.data.datasets[0].data = signal.map((y, i) => ({ x: time[i], y }));
    
    // Update individual wave datasets
    for (let i = 0; i < 4; i++) {
        const waveData = individualWaves.find(w => w.color === WAVE_COLORS[i]);
        if (waveData) {
            timeDomainChart.data.datasets[i + 1].data = waveData.data.map((y, j) => ({ x: time[j], y }));
            timeDomainChart.data.datasets[i + 1].hidden = false;
        } else {
            timeDomainChart.data.datasets[i + 1].data = [];
            timeDomainChart.data.datasets[i + 1].hidden = true;
        }
    }
    
    // Compute FFT (falls back to DFT if not power of 2)
    const dftResult = fft(signal);
    const magnitudes = computeMagnitudeSpectrum(dftResult);
    const phases = computePhaseSpectrum(dftResult);
    
    // Frequency bins
    const freqResolution = SAMPLING_RATE / numSamples;
    const frequencies = magnitudes.map((_, k) => k * freqResolution);
    const nyquistFreq = SAMPLING_RATE / 2;
    
    // Update Magnitude Spectrum
    // Only show frequencies up to Nyquist, and only show significant peaks
    const maxFreq = Math.min(50, nyquistFreq); // Limit display to 50 Hz for better visualization
    const filteredFreqIndices = frequencies
        .map((f, i) => ({ f, i }))
        .filter(item => item.f <= maxFreq);
    
    magnitudeChart.data.datasets[0].data = filteredFreqIndices.map(item => ({
        x: item.f,
        y: magnitudes[item.i]
    }));
    magnitudeChart.options.scales.x.max = maxFreq;
    
    // Update Phase Spectrum
    const significantPhases = filteredFreqIndices
        .filter(item => magnitudes[item.i] > 0.1) // Only show phase for significant frequencies
        .map(item => ({
            x: item.f,
            y: phases[item.i]
        }));
    
    phaseChart.data.datasets[0].data = significantPhases;
    phaseChart.options.scales.x.max = maxFreq;
    
    // Update charts
    timeDomainChart.update('none');
    magnitudeChart.update('none');
    phaseChart.update('none');
    
    // Update signal information
    updateSignalInfo(numSamples, freqResolution, signal, magnitudes);
}

function updateSignalInfo(numSamples, freqResolution, signal, magnitudes) {
    const nyquistFreq = SAMPLING_RATE / 2;
    const duration = numSamples / SAMPLING_RATE;
    
    // Calculate total power (Parseval's theorem)
    const totalPower = magnitudes.reduce((sum, mag, i) => {
        // DC and Nyquist components are not doubled
        const factor = (i === 0 || (i === magnitudes.length - 1 && numSamples % 2 === 0)) ? 1 : 0.5;
        return sum + factor * mag * mag;
    }, 0);
    
    // Calculate RMS amplitude
    const rmsAmplitude = Math.sqrt(signal.reduce((sum, val) => sum + val * val, 0) / signal.length);
    
    document.getElementById('samplingRate').textContent = `${SAMPLING_RATE} Hz`;
    document.getElementById('nyquistFreq').textContent = `${nyquistFreq} Hz`;
    document.getElementById('freqResolution').textContent = `${freqResolution.toFixed(3)} Hz`;
    document.getElementById('signalDuration').textContent = `${duration.toFixed(3)} s`;
    document.getElementById('totalPower').textContent = totalPower.toFixed(3);
    document.getElementById('rmsAmplitude').textContent = rmsAmplitude.toFixed(3);
}

// --- EVENT LISTENERS ---

function setupEventListeners() {
    // Wave enable checkboxes
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`enableWave${i}`).addEventListener('change', function() {
            const waveSection = this.closest('.wave-section');
            if (this.checked) {
                waveSection.classList.remove('disabled');
            } else {
                waveSection.classList.add('disabled');
            }
            updateVisualizations();
        });
    }
    
    // Wave parameter sliders
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`freq${i}`).addEventListener('input', function() {
            document.getElementById(`freq${i}Value`).textContent = `${this.value} Hz`;
            updateVisualizations();
        });
        
        document.getElementById(`amp${i}`).addEventListener('input', function() {
            document.getElementById(`amp${i}Value`).textContent = parseFloat(this.value).toFixed(1);
            updateVisualizations();
        });
        
        document.getElementById(`phase${i}`).addEventListener('input', function() {
            document.getElementById(`phase${i}Value`).textContent = `${this.value}°`;
            updateVisualizations();
        });
    }
    
    // Number of samples slider
    document.getElementById('numSamples').addEventListener('input', function() {
        document.getElementById('numSamplesValue').textContent = this.value;
        updateVisualizations();
    });
    
    // Noise toggle
    document.getElementById('addNoise').addEventListener('change', function() {
        const noiseLevelGroup = document.getElementById('noiseLevelGroup');
        const noiseLevelSlider = document.getElementById('noiseLevel');
        
        if (this.checked) {
            noiseLevelGroup.style.opacity = '1';
            noiseLevelSlider.disabled = false;
        } else {
            noiseLevelGroup.style.opacity = '0.4';
            noiseLevelSlider.disabled = true;
        }
        updateVisualizations();
    });
    
    // Noise level slider
    document.getElementById('noiseLevel').addEventListener('input', function() {
        document.getElementById('noiseLevelValue').textContent = parseFloat(this.value).toFixed(1);
        updateVisualizations();
    });
    
}

// --- MAIN INITIALIZATION ---

function main() {
    initCharts();
    setupEventListeners();
    
    // Make controls draggable on desktop
    if (window.innerWidth > 1200) {
        makeDraggable(document.getElementById('floatingControls'), document.getElementById('controlsTitle'));
    }
    
    // Initial visualization
    updateVisualizations();
}

window.addEventListener('load', main);
