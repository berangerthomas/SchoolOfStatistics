# School of Statistics

[![Documentation](https://img.shields.io/badge/Documentation-Sphinx-blue.svg)](https://berangerthomas.github.io/SchoolOfStatistics/)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/berangerthomas/schoolofstatistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Interactive visualizations for exploring statistical and machine learning concepts. Each page runs entirely in the browser (HTML, CSS, JavaScript with Chart.js) without requiring a server or build step.

## Available Pages

### Classification

- **[Direct Classification](direct_classifier.html)**: Generate synthetic 2D datasets and observe how class separation affects Gaussian Naive Bayes classifier performance. Displays ROC curve, AUC, confusion matrix, and standard metrics (accuracy, precision, recall, specificity, F1-score).

- **[Inverse Classification](inverse_classifier.html)**: Directly set confusion matrix values (TP, FP, TN, FN) and observe resulting metrics, ROC curve, and simulated score distributions. Parameters can be locked to constrain totals.

### Regression

- **[Linear Regression](linear_regression.html)**: Interactive point placement on canvas with linear or polynomial regression fitting. Displays residuals, coefficient of determination (R²), and regression diagnostics. Supports zoom, point dragging, and confidence band display.

### Signal Processing

- **[Fourier Transform](fourier_transform.html)**: Compose signals from sine waves and visualize their frequency spectrum. Up to 4 components with frequency, amplitude, and phase control. Displays time-domain signal, magnitude spectrum, phase spectrum, and signal metrics (sampling rate, Nyquist frequency, frequency resolution, total power, RMS).

## Project Structure

```
.
├── direct_classifier.html          # Direct classification (Naive Bayes)
├── inverse_classifier.html         # Inverse classification (confusion matrix)
├── linear_regression.html          # Linear/polynomial regression
├── fourier_transform.html          # Fourier transform
├── CHANGELOG.md                    # Version history
├── LICENSE
├── README.md
└── src/
    ├── assets/
    │   └── logo.jpg
    ├── css/
    │   ├── style.css               # Shared base styles
    │   ├── direct_classifier.css   # Page-specific styles
    │   ├── embedding_distances.css
    │   ├── fourier_transform.css
    │   └── linear_regression.css
    └── js/
        ├── common.js               # Shared utilities (metrics, ROC, matrices, drag, etc.)
        ├── direct_classifier.js
        ├── fourier_transform.js
        ├── inverse_classifier.js
        ├── linear_regression.js
        └── logistic_regression.js
```

## Usage

1. Clone the repository.
2. Open any `.html` file in your web browser.

No dependencies to install — all libraries are loaded via CDN.

## Versioning

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Roadmap

### Upcoming

- **Bias-Variance Tradeoff Explorer**: visualize bias-variance decomposition with polynomial fitting of increasing degree
- **k-Nearest Neighbors Playground**: interactive point placement and k-NN decision boundary visualization
- **Gradient Descent Visualizer**: real-time navigation on 2D loss surfaces, optimizer comparison
- **Principal Component Analysis (PCA) Step-by-Step**: Gaussian cloud generation and principal component visualization
- **Clustering Algorithms Visualizer**: k-Means and DBSCAN comparison on various dataset shapes
- **Neural Network Architecture & Forward Pass Visualizer**: layer-by-layer fully-connected network construction
- **Tokenization & Embedding Visualizer**: tokenization and 2D embedding space projection
- **Attention Mechanism Visualizer**: Transformer attention mechanism visualization
- **Probability Distributions Explorer**: exploration of standard distributions (Normal, Uniform, Exponential, Poisson, Binomial, Beta, Gamma, Chi-squared)
- **Markov Chain Text Generator**: Markov chain construction and text generation
- **A/B Testing Calculator**: statistical tool for hypothesis testing
- **Voice Signal Waveform Analyzer**: audio recording, waveform display, spectrogram computation, and dominant frequency identification

## License

See the [LICENSE](LICENSE) file.
