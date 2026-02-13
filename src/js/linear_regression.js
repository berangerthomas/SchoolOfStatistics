// ============================================================
// linear_regression.js — Linear Regression Playground
// ============================================================

// --- GLOBAL VARIABLES ---
let mainChart, residualChart;
let points = []; // Array of {x, y, id}
let nextPointId = 0;
let isDragging = false;
let draggedPointId = null;
let regressionCoefficients = null;

// Chart scales
const X_MIN = -10;
const X_MAX = 10;
const Y_MIN = -10;
const Y_MAX = 10;

// Zoom state
let currentXMin = X_MIN;
let currentXMax = X_MAX;
let currentYMin = Y_MIN;
let currentYMax = Y_MAX;
const ZOOM_FACTOR = 0.1; // 10% zoom per wheel step

// --- POLYNOMIAL REGRESSION IMPLEMENTATION ---

/**
 * Build Vandermonde matrix for polynomial regression
 * X: array of x values
 * degree: polynomial degree
 */
function buildVandermonde(X, degree) {
    const n = X.length;
    const V = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j <= degree; j++) {
            row.push(Math.pow(X[i], j));
        }
        V.push(row);
    }
    return V;
}

/**
 * Transpose a matrix
 */
function transpose(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = [];
    for (let j = 0; j < cols; j++) {
        const row = [];
        for (let i = 0; i < rows; i++) {
            row.push(matrix[i][j]);
        }
        result.push(row);
    }
    return result;
}

/**
 * Matrix multiplication: A * B
 */
function multiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;
    const result = [];
    for (let i = 0; i < rowsA; i++) {
        const row = [];
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                sum += A[i][k] * B[k][j];
            }
            row.push(sum);
        }
        result.push(row);
    }
    return result;
}

/**
 * Solve linear system Ax = b using Gaussian elimination
 */
function solveLinearSystem(A, b) {
    const n = A.length;
    // Augmented matrix
    const M = A.map((row, i) => [...row, b[i][0]]);
    
    // Forward elimination
    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
                maxRow = k;
            }
        }
        // Swap rows
        [M[i], M[maxRow]] = [M[maxRow], M[i]];
        
        // Make all rows below this one 0 in current column
        for (let k = i + 1; k < n; k++) {
            const factor = M[k][i] / M[i][i];
            for (let j = i; j <= n; j++) {
                M[k][j] -= factor * M[i][j];
            }
        }
    }
    
    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        x[i] = M[i][n] / M[i][i];
        for (let k = i - 1; k >= 0; k--) {
            M[k][n] -= M[k][i] * x[i];
        }
    }
    
    return x;
}

/**
 * Fit polynomial regression model
 * Returns coefficients [β0, β1, ..., βdegree]
 */
function fitPolynomialRegression(X, y, degree) {
    if (X.length < degree + 1) {
        return null; // Not enough points
    }
    
    const V = buildVandermonde(X, degree);
    const Vt = transpose(V);
    const VtV = multiply(Vt, V);
    const Vty = multiply(Vt, y.map(val => [val]));
    
    // Solve (V^T * V) * β = V^T * y
    const coefficients = solveLinearSystem(VtV, Vty);
    return coefficients;
}

/**
 * Predict y values for given x values using fitted model
 */
function predict(X, coefficients) {
    if (!coefficients) return X.map(() => 0);
    return X.map(x => {
        let y = 0;
        for (let i = 0; i < coefficients.length; i++) {
            y += coefficients[i] * Math.pow(x, i);
        }
        return y;
    });
}

/**
 * Calculate regression statistics
 */
function calculateStatistics(yTrue, yPred, n, degree) {
    const residuals = yTrue.map((y, i) => y - yPred[i]);
    const ssRes = residuals.reduce((sum, r) => sum + r * r, 0);
    const yMean = yTrue.reduce((sum, y) => sum + y, 0) / yTrue.length;
    const ssTot = yTrue.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
    
    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    const adjustedR2 = n > degree + 1 ? 1 - (1 - r2) * (n - 1) / (n - degree - 1) : r2;
    const mse = ssRes / n;
    const rmse = Math.sqrt(mse);
    const mae = residuals.reduce((sum, r) => sum + Math.abs(r), 0) / n;
    
    return { r2, adjustedR2, mse, rmse, mae, residuals };
}

/**
 * Calculate confidence intervals for predictions (pointwise ±2σ)
 */
function calculateConfidenceInterval(X, coefficients, residuals, confidence = 0.95) {
    if (!coefficients || X.length === 0) return X.map(() => ({ lower: 0, upper: 0 }));
    
    const n = residuals.length;
    const degree = coefficients.length - 1;
    
    // Estimate standard deviation of residuals
    const ssRes = residuals.reduce((sum, r) => sum + r * r, 0);
    const sigma = Math.sqrt(ssRes / Math.max(1, n - degree - 1));
    
    // For simplicity, use ±2σ as approximate 95% confidence interval
    const multiplier = 2;
    
    return X.map(x => {
        const yPred = predict([x], coefficients)[0];
        return {
            lower: yPred - multiplier * sigma,
            upper: yPred + multiplier * sigma
        };
    });
}

// --- CHART INITIALIZATION ---

function initCharts() {
    // Main chart (scatter + regression line)
    const mainCtx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(mainCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Data Points',
                    data: [],
                    backgroundColor: '#1976d2',
                    borderColor: '#1976d2',
                    borderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointHoverBorderWidth: 3
                },
                {
                    label: 'Regression Line',
                    data: [],
                    type: 'line',
                    borderColor: '#d32f2f',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Confidence Band Lower',
                    data: [],
                    type: 'line',
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(211, 47, 47, 0.1)',
                    borderWidth: 0,
                    pointRadius: 0,
                    fill: '+1'
                },
                {
                    label: 'Confidence Band Upper',
                    data: [],
                    type: 'line',
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(211, 47, 47, 0.1)',
                    borderWidth: 0,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Residuals',
                    data: [],
                    type: 'line',
                    borderColor: '#757575',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    showLine: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    min: X_MIN,
                    max: X_MAX,
                    title: { display: true, text: 'X' }
                },
                y: {
                    min: Y_MIN,
                    max: Y_MAX,
                    title: { display: true, text: 'Y' }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        filter: function(item) {
                            // Hide confidence band entries from legend
                            return item.text !== 'Confidence Band Lower' && item.text !== 'Confidence Band Upper';
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.label === 'Data Points') {
                                return `Point (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)})`;
                            }
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(3)}`;
                        }
                    }
                }
            }
            // Note: onClick and onHover removed - using native events instead
        }
    });

    // Residual chart
    const residualCtx = document.getElementById('residualChart').getContext('2d');
    residualChart = new Chart(residualCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Residuals',
                    data: [],
                    backgroundColor: '#1976d2',
                    pointRadius: 5
                },
                {
                    label: 'Zero Line',
                    data: [{ x: Y_MIN, y: 0 }, { x: Y_MAX, y: 0 }],
                    type: 'line',
                    borderColor: '#d32f2f',
                    borderWidth: 2,
                    pointRadius: 0,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    title: { display: true, text: 'Predicted Values' }
                },
                y: {
                    title: { display: true, text: 'Residuals' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// --- INTERACTION HANDLERS ---

function findNearestPoint(x, y, threshold = 1.0) {
    for (const point of points) {
        const dist = Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2));
        if (dist < threshold) {
            return point;
        }
    }
    return null;
}

function getMousePos(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

function getChartCoordinates(canvas, event) {
    const pos = getMousePos(canvas, event);
    return {
        x: mainChart.scales.x.getValueForPixel(pos.x),
        y: mainChart.scales.y.getValueForPixel(pos.y)
    };
}

function addPoint(x, y) {
    // Clamp values to current chart bounds (allows adding points outside initial view when zoomed out)
    x = Math.max(currentXMin, Math.min(currentXMax, x));
    y = Math.max(currentYMin, Math.min(currentYMax, y));
    
    points.push({ x, y, id: nextPointId++ });
    updateRegression();
}

function removePoint(id) {
    points = points.filter(p => p.id !== id);
    updateRegression();
}

function updatePointPosition(id, x, y) {
    const point = points.find(p => p.id === id);
    if (point) {
        point.x = Math.max(currentXMin, Math.min(currentXMax, x));
        point.y = Math.max(currentYMin, Math.min(currentYMax, y));
        updateRegression();
    }
}

// --- ZOOM FUNCTIONS ---

function zoom(factor, centerX, centerY) {
    const xRange = currentXMax - currentXMin;
    const yRange = currentYMax - currentYMin;
    
    // Calculate new ranges
    const newXRange = xRange * factor;
    const newYRange = yRange * factor;
    
    // Calculate ratios for center point
    const xRatio = (centerX - currentXMin) / xRange;
    const yRatio = (centerY - currentYMin) / yRange;
    
    // Calculate new bounds keeping the center point stable
    currentXMin = centerX - newXRange * xRatio;
    currentXMax = centerX + newXRange * (1 - xRatio);
    currentYMin = centerY - newYRange * yRatio;
    currentYMax = centerY + newYRange * (1 - yRatio);
    
    // Update chart scales
    mainChart.options.scales.x.min = currentXMin;
    mainChart.options.scales.x.max = currentXMax;
    mainChart.options.scales.y.min = currentYMin;
    mainChart.options.scales.y.max = currentYMax;
    mainChart.update('none');
}

function resetZoom() {
    if (points.length === 0) {
        // No points: reset to default view
        currentXMin = X_MIN;
        currentXMax = X_MAX;
        currentYMin = Y_MIN;
        currentYMax = Y_MAX;
    } else {
        // Fit to content: show all points with padding
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        
        const xMin = Math.min(...xs);
        const xMax = Math.max(...xs);
        const yMin = Math.min(...ys);
        const yMax = Math.max(...ys);
        
        // Add padding (20% of range or minimum 2 units)
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;
        const xPadding = Math.max(xRange * 0.1, 1);
        const yPadding = Math.max(yRange * 0.1, 1);
        
        currentXMin = xMin - xPadding;
        currentXMax = xMax + xPadding;
        currentYMin = yMin - yPadding;
        currentYMax = yMax + yPadding;
    }
    
    mainChart.options.scales.x.min = currentXMin;
    mainChart.options.scales.x.max = currentXMax;
    mainChart.options.scales.y.min = currentYMin;
    mainChart.options.scales.y.max = currentYMax;
    mainChart.update('none');
}

// --- UPDATE REGRESSION AND UI ---

function updateRegression() {
    const degree = parseInt(document.getElementById('degreeSlider').value);
    const showResiduals = document.getElementById('showResiduals').checked;
    const showConfidence = document.getElementById('showConfidence').checked;
    
    // Update main chart data points
    mainChart.data.datasets[0].data = points.map(p => ({ x: p.x, y: p.y }));
    
    if (points.length < degree + 1) {
        // Not enough points for regression
        mainChart.data.datasets[1].data = [];
        mainChart.data.datasets[2].data = [];
        mainChart.data.datasets[3].data = [];
        mainChart.data.datasets[4].data = [];
        residualChart.data.datasets[0].data = [];
        updateStatisticsDisplay({ r2: 0, adjustedR2: 0, mse: 0, rmse: 0, mae: 0 }, points.length);
        mainChart.update('none');
        residualChart.update('none');
        return;
    }
    
    // Fit regression model
    const X = points.map(p => p.x);
    const y = points.map(p => p.y);
    regressionCoefficients = fitPolynomialRegression(X, y, degree);
    
    // Generate smooth curve for visualization
    const xCurve = [];
    for (let x = currentXMin; x <= currentXMax; x += 0.2) {
        xCurve.push(x);
    }
    const yCurve = predict(xCurve, regressionCoefficients);
    
    // Update regression line
    mainChart.data.datasets[1].data = xCurve.map((x, i) => ({ x, y: yCurve[i] }));
    
    // Calculate predictions for actual points
    const yPred = predict(X, regressionCoefficients);
    const stats = calculateStatistics(y, yPred, points.length, degree);
    
    // Update confidence band
    if (showConfidence) {
        const confidenceIntervals = calculateConfidenceInterval(xCurve, regressionCoefficients, stats.residuals);
        mainChart.data.datasets[2].data = xCurve.map((x, i) => ({ x, y: confidenceIntervals[i].lower }));
        mainChart.data.datasets[3].data = xCurve.map((x, i) => ({ x, y: confidenceIntervals[i].upper }));
    } else {
        mainChart.data.datasets[2].data = [];
        mainChart.data.datasets[3].data = [];
    }
    
    // Update residual lines
    if (showResiduals) {
        const residualLines = [];
        for (let i = 0; i < points.length; i++) {
            residualLines.push({ x: X[i], y: y[i] });
            residualLines.push({ x: X[i], y: yPred[i] });
            residualLines.push({ x: NaN, y: NaN }); // Break line
        }
        mainChart.data.datasets[4].data = residualLines;
    } else {
        mainChart.data.datasets[4].data = [];
    }
    
    // Update residual chart
    residualChart.data.datasets[0].data = yPred.map((yp, i) => ({ x: yp, y: stats.residuals[i] }));
    
    // Update statistics display
    updateStatisticsDisplay(stats, points.length);
    
    mainChart.update('none');
    residualChart.update('none');
}

function updateStatisticsDisplay(stats, n) {
    document.getElementById('r2Value').textContent = n > 0 ? stats.r2.toFixed(4) : '-';
    document.getElementById('adjustedR2Value').textContent = n > 0 ? stats.adjustedR2.toFixed(4) : '-';
    document.getElementById('mseValue').textContent = n > 0 ? stats.mse.toFixed(4) : '-';
    document.getElementById('maeValue').textContent = n > 0 ? stats.mae.toFixed(4) : '-';
    document.getElementById('rmseValue').textContent = n > 0 ? stats.rmse.toFixed(4) : '-';
    document.getElementById('nPointsValue').textContent = n;
}

// --- EVENT LISTENERS ---

function setupEventListeners() {
    // Degree slider
    const degreeSlider = document.getElementById('degreeSlider');
    degreeSlider.addEventListener('input', function() {
        document.getElementById('degreeValue').textContent = this.value;
        updateRegression();
    });
    
    // Show residuals toggle
    document.getElementById('showResiduals').addEventListener('change', updateRegression);
    
    // Show confidence toggle
    document.getElementById('showConfidence').addEventListener('change', updateRegression);
    
    // Clear points button
    document.getElementById('clearPointsBtn').addEventListener('click', function() {
        points = [];
        updateRegression();
    });
    
    // Add random points button
    document.getElementById('addRandomBtn').addEventListener('click', function() {
        for (let i = 0; i < 5; i++) {
            const x = currentXMin + Math.random() * (currentXMax - currentXMin);
            const y = (currentYMin + Math.random() * (currentYMax - currentYMin)) + randomGaussian(0, 2);
            addPoint(x, y);
        }
    });
    
    // Reset zoom button
    document.getElementById('resetZoomBtn').addEventListener('click', resetZoom);
    
    // Canvas interactions
    const canvas = document.getElementById('mainChart');
    let hasDragged = false;
    let clickedOnPoint = false;
    
    // Canvas for zoom with mouse wheel
    canvas.addEventListener('wheel', function(event) {
        event.preventDefault();
        
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        
        // Get chart coordinates of mouse position
        const centerX = mainChart.scales.x.getValueForPixel(mouseX);
        const centerY = mainChart.scales.y.getValueForPixel(mouseY);
        
        // Zoom in (scroll up, deltaY < 0) or zoom out (scroll down, deltaY > 0)
        const zoomDirection = event.deltaY < 0 ? (1 - ZOOM_FACTOR) : (1 + ZOOM_FACTOR);
        zoom(zoomDirection, centerX, centerY);
    }, { passive: false });
    
    // Mouse down - start drag or prepare for click
    canvas.addEventListener('mousedown', function(event) {
        if (event.button !== 0) return; // Only left click
        
        event.preventDefault();
        event.stopPropagation();
        
        const coords = getChartCoordinates(canvas, event);
        const nearestPoint = findNearestPoint(coords.x, coords.y);
        
        hasDragged = false;
        
        if (nearestPoint) {
            // Clicked on existing point - start dragging
            isDragging = true;
            draggedPointId = nearestPoint.id;
            clickedOnPoint = true;
            canvas.classList.add('dragging');
        } else {
            // Clicked on empty space - prepare for adding point
            clickedOnPoint = false;
        }
    });
    
    // Mouse move - handle dragging
    document.addEventListener('mousemove', function(event) {
        // Update cursor style on hover when not dragging
        if (!isDragging) {
            const coords = getChartCoordinates(canvas, event);
            if (coords.x >= X_MIN && coords.x <= X_MAX && coords.y >= Y_MIN && coords.y <= Y_MAX) {
                const nearestPoint = findNearestPoint(coords.x, coords.y);
                if (nearestPoint) {
                    canvas.classList.add('hover-point');
                } else {
                    canvas.classList.remove('hover-point');
                }
            }
            return;
        }
        
        // We are dragging
        hasDragged = true;
        
        const rect = canvas.getBoundingClientRect();
        let canvasX = event.clientX - rect.left;
        let canvasY = event.clientY - rect.top;
        
        // Clamp to canvas bounds
        canvasX = Math.max(0, Math.min(canvasX, rect.width));
        canvasY = Math.max(0, Math.min(canvasY, rect.height));
        
        const x = mainChart.scales.x.getValueForPixel(canvasX);
        const y = mainChart.scales.y.getValueForPixel(canvasY);
        
        updatePointPosition(draggedPointId, x, y);
    });
    
    // Mouse up - end drag or handle click
    document.addEventListener('mouseup', function(event) {
        if (event.button !== 0) return; // Only left click
        
        if (isDragging) {
            // End drag
            isDragging = false;
            draggedPointId = null;
            canvas.classList.remove('dragging');
        } else if (!clickedOnPoint) {
            // This was a click on empty space - add point
            const coords = getChartCoordinates(canvas, event);
            // Check if click is within current visible bounds (not just initial bounds)
            if (coords.x >= currentXMin && coords.x <= currentXMax && coords.y >= currentYMin && coords.y <= currentYMax) {
                addPoint(coords.x, coords.y);
            }
        }
        
        // Reset flags
        hasDragged = false;
        clickedOnPoint = false;
    });
    
    // Right-click to remove point
    canvas.addEventListener('contextmenu', function(event) {
        event.preventDefault();
        event.stopPropagation();
        
        if (isDragging) {
            // Cancel drag
            isDragging = false;
            draggedPointId = null;
            canvas.classList.remove('dragging');
            return;
        }
        
        const coords = getChartCoordinates(canvas, event);
        const clickedPoint = findNearestPoint(coords.x, coords.y);
        if (clickedPoint) {
            removePoint(clickedPoint.id);
        }
    });
    
    // Prevent context menu on right-click
    canvas.addEventListener('contextmenu', function(event) {
        event.preventDefault();
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
    
    // Add some initial random points
    for (let i = 0; i < 8; i++) {
        const x = X_MIN + Math.random() * (X_MAX - X_MIN);
        const y = 2 + 0.5 * x + randomGaussian(0, 1.5);
        points.push({ x, y, id: nextPointId++ });
    }
    
    updateRegression();
}

window.addEventListener('load', main);
