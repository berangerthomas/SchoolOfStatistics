// ============================================================
// common.js — Shared utilities for School of Statistics
// ============================================================

// --- METRIC EXPLANATIONS (used in tooltip callbacks) ---
const metricExplanations = {
    'AUC': {
        description: "Probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.",
        range: "0 to 1. 0.5 corresponds to random chance.",
        formula: "Area Under the ROC Curve"
    },
    'Accuracy': {
        description: "Proportion of all predictions that are correct.",
        range: "0 to 1.",
        formula: "(TP + TN) / (TP + TN + FP + FN)"
    },
    'Precision': {
        description: "Proportion of positive predictions that are correct. A high value indicates a low false positive rate.",
        range: "0 to 1.",
        formula: "TP / (TP + FP)"
    },
    'Recall': {
        description: "Proportion of actual positives correctly identified. Also called Sensitivity or True Positive Rate.",
        range: "0 to 1.",
        formula: "TP / (TP + FN)"
    },
    'Specificity': {
        description: "Proportion of actual negatives correctly identified. Also called True Negative Rate.",
        range: "0 to 1.",
        formula: "TN / (TN + FP)"
    },
    'F1-Score': {
        description: "Harmonic mean of Precision and Recall. Balances both metrics in a single value.",
        range: "0 to 1.",
        formula: "2 * (Precision * Recall) / (Precision + Recall)"
    }
};

// --- RANDOM GAUSSIAN (Box-Muller transform) ---
function randomGaussian(mean = 0, stdDev = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// --- ROC CURVE & AUC CALCULATION ---
function calculateRocAndAuc(labels, scores) {
    const pairs = labels.map((label, i) => ({ label, score: scores[i] }));
    pairs.sort((a, b) => b.score - a.score);
    let tp = 0, fp = 0;
    const total_pos = labels.filter(l => l === 1).length;
    const total_neg = labels.length - total_pos;
    if (total_pos === 0 || total_neg === 0) {
        return { rocPoints: [{ x: 0, y: 0 }, { x: 1, y: 1 }], auc: 0.5 };
    }
    const rocPoints = [{ x: 0, y: 0 }];
    let auc = 0, prev_tpr = 0, prev_fpr = 0;
    for (const pair of pairs) {
        if (pair.label === 1) tp++; else fp++;
        const tpr = total_pos > 0 ? tp / total_pos : 0;
        const fpr = total_neg > 0 ? fp / total_neg : 0;
        auc += (tpr + prev_tpr) / 2 * (fpr - prev_fpr);
        rocPoints.push({ x: fpr, y: tpr });
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    return { rocPoints, auc };
}

// --- CONFUSION MATRIX DRAWING (Canvas 2D) ---
function drawConfusionMatrix(canvasId, tp, fp, tn, fn) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const margin = 50;
    const gridW = w - margin, gridH = h - margin;
    const cellW = gridW / 2, cellH = gridH / 2;
    const max_val = Math.max(tp, fp, tn, fn);
    const baseColor = [8, 48, 107];

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
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = 'bold 20px Segoe UI';
        ctx.fillText(cell.label, margin + cell.x + cellW / 2, cell.y + cellH / 2 - 12);
        ctx.font = '18px Segoe UI';
        ctx.fillText(cell.value, margin + cell.x + cellW / 2, cell.y + cellH / 2 + 12);
    });

    ctx.fillStyle = '#333';
    ctx.font = 'bold 14px Segoe UI';
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

// --- CUSTOM DATALABELS PLUGIN (for bar charts) ---
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

// --- DEBOUNCE UTILITY ---
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// --- DRAGGABLE CONTROLS PANEL ---
function makeDraggable(element, handle) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    let isDragging = false;
    
    handle.onmousedown = (e) => {
        // Don't drag if clicking on a form control (slider, input, etc.)
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || 
            e.target.tagName === 'TEXTAREA' || e.target.classList.contains('slider')) {
            return;
        }
        
        isDragging = true;
        pos3 = e.clientX;
        pos4 = e.clientY;
        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
    };
    
    const elementDrag = (e) => {
        if (!isDragging) return;
        e.preventDefault();
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;
        element.style.top = (element.offsetTop - pos2) + "px";
        element.style.left = (element.offsetLeft - pos1) + "px";
    };
    
    const closeDragElement = () => {
        isDragging = false;
        document.onmouseup = null;
        document.onmousemove = null;
    };
}

// --- METRICS BAR CHART TOOLTIP CALLBACK ---
function metricsTooltipCallback(context) {
    const label = context.chart.data.labels[context.dataIndex];
    const value = context.raw.toFixed(3);
    const explanation = metricExplanations[label];
    let tooltipText = [`${label}: ${value}`];
    if (explanation) {
        tooltipText.push('');
        const roleLines = `${explanation.description}`.match(/.{1,50}(\s|$)/g) || [];
        roleLines.forEach(line => tooltipText.push(line.trim()));
        tooltipText.push(`Range: ${explanation.range}`);
        tooltipText.push(`Formula: ${explanation.formula}`);
    }
    return tooltipText;
}
