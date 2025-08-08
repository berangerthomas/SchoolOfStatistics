// --- GLOBAL VARIABLES ---
let dataChart, rocChart, metricsChart;
const N_SAMPLES_PER_CLASS = 100;

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

// --- DATA GENERATION ---
function randomGaussian(mean = 0, stdDev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateData(separation, stdDev) {
    const data = [], labels = [];
    for (let i = 0; i < N_SAMPLES_PER_CLASS; i++) { data.push({ x: randomGaussian(-separation / 2, stdDev), y: randomGaussian(0, stdDev) }); labels.push(0); }
    for (let i = 0; i < N_SAMPLES_PER_CLASS; i++) { data.push({ x: randomGaussian(separation / 2, stdDev), y: randomGaussian(0, stdDev) }); labels.push(1); }
    return { data, labels };
}

// --- CLASSIFIER: GAUSSIAN NAIVE BAYES ---
class GaussianNB {
    fit(X, y) {
        const classes = [...new Set(y)];
        this.classes = classes;
        this.params = {};
        for (const cls of classes) {
            const X_cls = X.filter((_, i) => y[i] === cls);
            const mean_x = X_cls.reduce((a, b) => a + b.x, 0) / X_cls.length;
            const mean_y = X_cls.reduce((a, b) => a + b.y, 0) / X_cls.length;
            this.params[cls] = {
                prior: X_cls.length / X.length,
                mean: [mean_x, mean_y],
                variance: [Math.max(1e-9, X_cls.reduce((a, b) => a + Math.pow(b.x - mean_x, 2), 0) / X_cls.length), Math.max(1e-9, X_cls.reduce((a, b) => a + Math.pow(b.y - mean_y, 2), 0) / X_cls.length)]
            };
        }
    }
    _pdf(x, mean, variance) { const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance)); return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent; }
    predict_proba(X) {
        return X.map(point => {
            const posteriors = {};
            for (const cls of this.classes) {
                const prior = Math.log(this.params[cls].prior);
                const likelihood_x = Math.log(this._pdf(point.x, this.params[cls].mean[0], this.params[cls].variance[0]));
                const likelihood_y = Math.log(this._pdf(point.y, this.params[cls].mean[1], this.params[cls].variance[1]));
                posteriors[cls] = prior + likelihood_x + likelihood_y;
            }
            const max_posterior = Math.max(...Object.values(posteriors));
            const exps = Object.fromEntries(Object.entries(posteriors).map(([k, v]) => [k, Math.exp(v - max_posterior)]));
            const sum_exps = Object.values(exps).reduce((a, b) => a + b);
            return exps[1] / sum_exps;
        });
    }
}

// --- METRICS CALCULATIONS ---
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

function getConfusionMatrix(labels, scores, threshold) {
    let vp = 0, fp = 0, vn = 0, fn = 0;
    labels.forEach((label, i) => {
        const prediction = scores[i] >= threshold ? 1 : 0;
        if (prediction === 1 && label === 1) vp++;
        else if (prediction === 1 && label === 0) fp++;
        else if (prediction === 0 && label === 0) vn++;
        else if (prediction === 0 && label === 1) fn++;
    });
    return { vp, fp, vn, fn };
}

function drawConfusionMatrix(canvasId, vp, fp, vn, fn) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const margin = 50, gridW = w - margin, gridH = h - margin, cellW = gridW / 2, cellH = gridH / 2;
    const max_val = Math.max(vp, fp, vn, fn);
    const baseColor = [8, 48, 107];
    const cells = [{ label: 'TN', value: vn, x: 0, y: cellH }, { label: 'FP', value: fp, x: cellW, y: cellH }, { label: 'FN', value: fn, x: 0, y: 0 }, { label: 'TP', value: vp, x: cellW, y: 0 }];
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

// --- UI UPDATE ---
function updateApplication() {
    const separation = parseFloat(document.getElementById('separationSlider').value);
    const stdDev = parseFloat(document.getElementById('stdDevSlider').value);
    document.getElementById('separationValue').textContent = separation.toFixed(1);
    document.getElementById('stdDevValue').textContent = stdDev.toFixed(1);

    const { data, labels } = generateData(separation, stdDev);
    const model = new GaussianNB();
    model.fit(data, labels);
    const scores = model.predict_proba(data);
    const { rocPoints, auc } = calculateRocAndAuc(labels, scores);
    const { vp, fp, vn, fn } = getConfusionMatrix(labels, scores, 0.5);
    const total = vp + fp + vn + fn;
    const precision = (vp + fp) > 0 ? vp / (vp + fp) : 0;
    const recall = (vp + fn) > 0 ? vp / (vp + fn) : 0;
    const specificity = (vn + fp) > 0 ? vn / (vn + fp) : 0;
    const f1score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = total > 0 ? (vp + vn) / total : 0;

    drawConfusionMatrix('matrixChart', vp, fp, vn, fn);

    dataChart.data.datasets[0].data = data.filter((_, i) => labels[i] === 0);
    dataChart.data.datasets[1].data = data.filter((_, i) => labels[i] === 1);
    dataChart.update('none');

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
    const dataCtx = document.getElementById('dataChart').getContext('2d');
    dataChart = new Chart(dataCtx, { type: 'scatter', data: { datasets: [{ label: 'Negative Class', data: [], backgroundColor: '#0D47A1' }, { label: 'Positive Class', data: [], backgroundColor: '#B71C1C' }] }, options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 } } });
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

function makeDraggable(element, handle) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    handle.onmousedown = (e) => { e.preventDefault(); pos3 = e.clientX; pos4 = e.clientY; document.onmouseup = closeDragElement; document.onmousemove = elementDrag; };
    const elementDrag = (e) => { e.preventDefault(); pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY; pos3 = e.clientX; pos4 = e.clientY; element.style.top = (element.offsetTop - pos2) + "px"; element.style.left = (element.offsetLeft - pos1) + "px"; };
    const closeDragElement = () => { document.onmouseup = null; document.onmousemove = null; };
}

window.addEventListener('load', function () {
    initCharts();
    const sliders = ['separationSlider', 'stdDevSlider'];
    sliders.forEach(id => { document.getElementById(id).addEventListener('input', updateApplication); });
    if (window.innerWidth > 1200) { makeDraggable(document.getElementById('floatingControls'), document.getElementById('controlsTitle')); }
    updateApplication();
});
