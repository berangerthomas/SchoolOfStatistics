// --- GLOBAL VARIABLES ---
let scoresChart, rocChart, metricsChart;
const metricExplanations = {
    'AUC': {
        description: "Measures the model's ability to distinguish between positive and negative classes. It represents the probability that a random positive instance is ranked higher than a random negative instance.",
        range: "Ranges from 0 (worst) to 1 (best). 0.5 is random chance.",
        formula: "Area Under the ROC Curve"
    },
    'Accuracy': {
        description: "The proportion of all predictions that are correct. It's a general measure of the model's performance.",
        range: "Ranges from 0 (worst) to 1 (best).",
        formula: "(TP + TN) / (TP + TN + FP + FN)"
    },
    'Precision': {
        description: "Of all the positive predictions made by the model, how many were actually positive. High precision indicates a low false positive rate.",
        range: "Ranges from 0 (worst) to 1 (best).",
        formula: "TP / (TP + FP)"
    },
    'Recall': {
        description: "Of all the actual positive instances, how many did the model correctly identify. Also known as Sensitivity or True Positive Rate.",
        range: "Ranges from 0 (worst) to 1 (best).",
        formula: "TP / (TP + FN)"
    },
    'Specificity': {
        description: "Of all the actual negative instances, how many did the model correctly identify. Also known as True Negative Rate.",
        range: "Ranges from 0 (worst) to 1 (best).",
        formula: "TN / (TN + FP)"
    },
    'F1-Score': {
        description: "The harmonic mean of Precision and Recall. It provides a single score that balances both concerns, useful for imbalanced classes.",
        range: "Ranges from 0 (worst) to 1 (best).",
        formula: "2 * (Precision * Recall) / (Precision + Recall)"
    }
};
let lockState = { tp: false, fp: false, tn: false, fn: false };
let currentState = { tp: 70, fp: 20, tn: 80, fn: 30 };
const TOTAL_SAMPLES = 200;

// --- SIMULATION & CALCULATIONS ---
function randomGaussian(mean = 0, stdDev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

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

function calculateRocAndAuc(labels, scores) {
    const pairs = labels.map((label, i) => ({ label, score: scores[i] }));
    pairs.sort((a, b) => b.score - a.score);
    let tp = 0, fp = 0;
    const total_pos = labels.filter(l => l === 1).length;
    const total_neg = labels.length - total_pos;
    if (total_pos === 0 || total_neg === 0) return { rocPoints: [{ x: 0, y: 0 }, { x: 1, y: 1 }], auc: 0.5 };
    const rocPoints = [{ x: 0, y: 0 }];
    let auc = 0, prev_tpr = 0, prev_fpr = 0;
    for (const pair of pairs) {
        if (pair.label === 1) tp++; else fp++;
        const tpr = tp / total_pos;
        const fpr = fp / total_neg;
        auc += (tpr + prev_tpr) / 2 * (fpr - prev_fpr);
        rocPoints.push({ x: fpr, y: tpr });
        prev_tpr = tpr; prev_fpr = fpr;
    }
    return { rocPoints, auc };
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

// [CORRIGÉ] Fonction de dessin utilisant la palette "Blues"
function drawConfusionMatrix(canvasId, tp, fp, tn, fn) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const margin = 50, gridW = w - margin, gridH = h - margin, cellW = gridW / 2, cellH = gridH / 2;
    const max_val = Math.max(tp, fp, tn, fn);
    const baseColor = [8, 48, 107]; // "Blues" palette base color
    const cells = [
        { label: 'TN', value: tn, x: 0, y: cellH },
        { label: 'FP', value: fp, x: cellW, y: cellH },
        { label: 'FN', value: fn, x: 0, y: 0 },
        { label: 'TP', value: tp, x: cellW, y: 0 }
    ];
    cells.forEach(cell => {
        const intensity = max_val > 0 ? cell.value / max_val : 0;
        ctx.fillStyle = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, ${intensity})`;
        ctx.fillRect(margin + cell.x, cell.y, cellW, cellH);
        ctx.fillStyle = intensity > 0.5 ? 'white' : 'black';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.font = 'bold 20px Segoe UI'; ctx.fillText(cell.label, margin + cell.x + cellW / 2, cell.y + cellH / 2 - 12);
        ctx.font = '18px Segoe UI'; ctx.fillText(cell.value, margin + cell.x + cellW / 2, cell.y + cellH / 2 + 12);
    });
    ctx.fillStyle = '#333'; ctx.font = 'bold 14px Segoe UI';
    ctx.fillText('Negative', margin + cellW / 2, gridH + 20);
    ctx.fillText('Positive', margin + cellW + cellW / 2, gridH + 20);
    ctx.save();
    ctx.translate(20, gridH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Positive', -cellH / 2, 0);
    ctx.fillText('Negative', cellH / 2, 0);
    ctx.restore();
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
const customDatalabelsPlugin = {
    id: 'customDatalabels',
    afterDatasetsDraw: (chart) => {
        const ctx = chart.ctx;
        ctx.save();
        ctx.font = 'bold 12px Segoe UI';
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        chart.data.datasets.forEach((dataset, i) => {
            const meta = chart.getDatasetMeta(i);
            meta.data.forEach((bar, index) => {
                const data = dataset.data[index];
                if (bar.height > 15) {
                    ctx.textBaseline = 'bottom';
                    ctx.fillText(data.toFixed(3), bar.x, bar.y + bar.height - 5);
                }
            });
        });
        ctx.restore();
    }
};

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
                        label: function (context) {
                            const label = context.chart.data.labels[context.dataIndex];
                            const value = context.raw.toFixed(3);
                            const explanation = metricExplanations[label];
                            let tooltipText = [`${label}: ${value}`];
                            if (explanation) {
                                tooltipText.push('');
                                const roleLines = `Role: ${explanation.description}`.match(/.{1,50}(\s|$)/g) || [];
                                roleLines.forEach(line => tooltipText.push(line.trim()));
                                tooltipText.push(`Range: ${explanation.range}`);
                                tooltipText.push(`Formula: ${explanation.formula}`);
                            }
                            return tooltipText;
                        }
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

function makeDraggable(element, handle) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    handle.onmousedown = (e) => { e.preventDefault(); pos3 = e.clientX; pos4 = e.clientY; document.onmouseup = closeDragElement; document.onmousemove = elementDrag; };
    const elementDrag = (e) => { e.preventDefault(); pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY; pos3 = e.clientX; pos4 = e.clientY; element.style.top = (element.offsetTop - pos2) + "px"; element.style.left = (element.offsetLeft - pos1) + "px"; };
    const closeDragElement = () => { document.onmouseup = null; document.onmousemove = null; };
}

window.addEventListener('load', function () {
    initCharts();
    Object.keys(currentState).forEach(param => {
        const slider = document.getElementById(`${param}Slider`);
        slider.addEventListener('input', () => document.getElementById(`${param}Value`).textContent = slider.value);
        slider.addEventListener('change', () => adjustValues(param, parseInt(slider.value)));
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
