// --- GLOBAL VARIABLES ---
let dataChart, rocChart, metricsChart;
const N_SAMPLES_PER_CLASS = 100;

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
function getConfusionMatrix(labels, scores, threshold) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    labels.forEach((label, i) => {
        const prediction = scores[i] >= threshold ? 1 : 0;
        if (prediction === 1 && label === 1) tp++;
        else if (prediction === 1 && label === 0) fp++;
        else if (prediction === 0 && label === 0) tn++;
        else if (prediction === 0 && label === 1) fn++;
    });
    return { tp, fp, tn, fn };
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
    const { tp, fp, tn, fn } = getConfusionMatrix(labels, scores, 0.5);
    const total = tp + fp + tn + fn;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0;
    const f1score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = total > 0 ? (tp + tn) / total : 0;

    drawConfusionMatrix('matrixChart', tp, fp, tn, fn);

    dataChart.data.datasets[0].data = data.filter((_, i) => labels[i] === 0);
    dataChart.data.datasets[1].data = data.filter((_, i) => labels[i] === 1);
    dataChart.update('none');

    rocChart.data.datasets[0].data = rocPoints;
    rocChart.update('none');

    metricsChart.data.datasets[0].data = [auc, accuracy, precision, recall, specificity, f1score];
    metricsChart.update('none');
}

// --- INITIALIZATION ---
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
                        label: metricsTooltipCallback
                    }
                }
            },
            scales: { y: { beginAtZero: true, max: 1 } }
        }
    });
}

window.addEventListener('load', function () {
    initCharts();
    const sliders = ['separationSlider', 'stdDevSlider'];
    sliders.forEach(id => { document.getElementById(id).addEventListener('input', updateApplication); });
    if (window.innerWidth > 1200) { makeDraggable(document.getElementById('floatingControls'), document.getElementById('controlsTitle')); }
    updateApplication();
});
