// --- GLOBAL VARIABLES ---
let scoresChart, rocChart, metricsChart;
let lockState = { tp: false, fp: false, tn: false, fn: false };
let currentState = { tp: 40, fp: 10, tn: 45, fn: 5 };
const TOTAL_SAMPLES = 100;

// --- SIMULATION ---
function generateScoresFromMatrix(tp, fp, tn, fn) {
    const total_pos_real = tp + fn;
    const total_neg_real = tn + fp;
    if (total_pos_real === 0 || total_neg_real === 0) return { scores: [], labels: [] };
    const tpr = tp / total_pos_real;
    const tnr = tn / total_neg_real;
    const mean_pos = 0.5 + (tpr - 0.5);
    const mean_neg = 0.5 - (tnr - 0.5);
    const stdDev = 0.15;
    const scores = [], labels = [];
    for (let i = 0; i < total_neg_real; i++) { scores.push(randomGaussian(mean_neg, stdDev)); labels.push(0); }
    for (let i = 0; i < total_pos_real; i++) { scores.push(randomGaussian(mean_pos, stdDev)); labels.push(1); }
    return { scores, labels };
}

function createHistogramData(scores, labels, n_bins = 20) {
    const bins = Array(n_bins).fill(0).map(() => ({ pos: 0, neg: 0 }));
    const bin_labels = Array(n_bins).fill(0).map((_, i) => (i / n_bins).toFixed(2));
    scores.forEach((score, i) => {
        let bin_index = Math.floor(score * n_bins);
        if (bin_index < 0) bin_index = 0;
        if (bin_index >= n_bins) bin_index = n_bins - 1;
        if (labels[i] === 1) bins[bin_index].pos++;
        else bins[bin_index].neg++;
    });
    return { labels: bin_labels, pos_data: bins.map(b => b.pos), neg_data: bins.map(b => b.neg) };
}

// --- UI MANAGEMENT ---
function adjustValues(changedParam, newValue) {
    let values = { ...currentState };
    values[changedParam] = newValue;
    let diff = Object.values(values).reduce((sum, v) => sum + v, 0) - TOTAL_SAMPLES;
    const unlockedParams = Object.keys(values).filter(p => p !== changedParam && !lockState[p]);
    while (diff !== 0 && unlockedParams.length > 0) {
        let adjustedInLoop = false;
        if (diff > 0) {
            unlockedParams.sort((a, b) => values[b] - values[a]);
            for (const param of unlockedParams) { if (values[param] > 0) { values[param]--; diff--; adjustedInLoop = true; break; } }
        } else {
            unlockedParams.sort((a, b) => values[a] - values[b]);
            for (const param of unlockedParams) { if (values[param] < TOTAL_SAMPLES) { values[param]++; diff++; adjustedInLoop = true; break; } }
        }
        if (!adjustedInLoop) break;
    }
    if (diff === 0) { currentState = { ...values }; }
    updateUI();
}

function updateUI() {
    const { tp, fp, tn, fn } = currentState;
    const total = tp + fp + tn + fn;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0;
    const f1score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const { scores, labels } = generateScoresFromMatrix(tp, fp, tn, fn);
    const { rocPoints, auc } = calculateRocAndAuc(labels, scores);
    const histogram = createHistogramData(scores, labels);

    for (const param in currentState) {
        document.getElementById(`${param}Slider`).value = currentState[param];
        document.getElementById(`${param}Value`).textContent = currentState[param];
    }

    drawConfusionMatrix('matrixChart', tp, fp, tn, fn);

    scoresChart.data.labels = histogram.labels;
    scoresChart.data.datasets[0].data = histogram.neg_data;
    scoresChart.data.datasets[1].data = histogram.pos_data;
    scoresChart.update('none');

    rocChart.data.datasets[0].data = rocPoints;
    rocChart.update('none');

    metricsChart.data.datasets[0].data = [auc, accuracy, precision, recall, specificity, f1score];
    metricsChart.update('none');
}

// --- INITIALIZATION ---
function initCharts() {
    const scoresCtx = document.getElementById('scoresChart').getContext('2d');
    scoresChart = new Chart(scoresCtx, { type: 'bar', data: { labels: [], datasets: [{ label: 'Scores (Negative Class)', data: [], backgroundColor: '#0D47A1', barPercentage: 1.0, categoryPercentage: 1.0 }, { label: 'Scores (Positive Class)', data: [], backgroundColor: '#B71C1C', barPercentage: 1.0, categoryPercentage: 1.0 }] }, options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 }, scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: 'Number of Samples' } } } } });
    const rocCtx = document.getElementById('rocChart').getContext('2d');
    rocChart = new Chart(rocCtx, { type: 'scatter', data: { datasets: [{ label: 'ROC Curve', data: [], borderColor: '#0D47A1', backgroundColor: 'transparent', showLine: true, pointRadius: 0, borderWidth: 3 }, { label: 'Chance Line', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#666', showLine: true, pointRadius: 0, borderDash: [5, 5] }] }, options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 }, scales: { x: { min: 0, max: 1, title: { display: true, text: 'False Positive Rate' } }, y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' } } } } });

    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: { labels: ['AUC', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'], datasets: [{ data: [], backgroundColor: ['#673AB7', '#009688', '#1E88E5', '#388E3C', '#FB8C00', '#9C27B0'] }] },
        plugins: [customDatalabelsPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'x',
            animation: { duration: 0 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#000',
                    bodyColor: '#000',
                    borderColor: '#555',
                    borderWidth: 1,
                    padding: 15,
                    displayColors: false,
                    callbacks: {
                        label: metricsTooltipCallback
                    }
                }
            },
            scales: { y: { beginAtZero: true, max: 1 } }
        }
    });
}

function updateSliderDisabledState() {
    const sliders = { tp: document.getElementById('tpSlider'), fp: document.getElementById('fpSlider'), tn: document.getElementById('tnSlider'), fn: document.getElementById('fnSlider') };
    const lockedCount = Object.values(lockState).filter(isLocked => isLocked).length;
    for (const param in lockState) { sliders[param].disabled = lockState[param] || lockedCount >= 3; }
}

window.addEventListener('load', function () {
    initCharts();
    Object.keys(currentState).forEach(param => {
        const slider = document.getElementById(`${param}Slider`);
        slider.addEventListener('input', () => {
            document.getElementById(`${param}Value`).textContent = slider.value;
            adjustValues(param, parseInt(slider.value));
        });
    });
    document.querySelectorAll('.lock-toggle').forEach(lock => {
        lock.addEventListener('click', function () {
            const param = this.dataset.param;
            lockState[param] = !lockState[param];
            this.textContent = lockState[param] ? '🔒' : '🔓';
            this.classList.toggle('locked', lockState[param]);
            updateSliderDisabledState();
        });
    });
    if (window.innerWidth > 1200) { makeDraggable(document.getElementById('floatingControls'), document.getElementById('controlsTitle')); }
    updateSliderDisabledState();
    updateUI();
});
