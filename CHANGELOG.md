# Changelog

## [0.2.0] - 2026-01-13

### Added
- Fourier Transform Visualizer: compose signals from up to 4 sine waves with frequency, amplitude, and phase control; displays time-domain signal, magnitude spectrum, phase spectrum, and signal metrics (sampling rate, Nyquist frequency, frequency resolution, total power, RMS amplitude)
- Linear regression playground: interactive point placement with linear and polynomial regression (degrees 1-10); features residual visualization, confidence bands (±2σ), zoom and pan, drag-and-drop points, and regression statistics (R², adjusted R², MSE, MAE, RMSE, residual plot)

### Changed
- Refactored shared utilities in `common.js` for matrix operations, statistical calculations, and chart management

## [0.1.0] - 2025-08-15

### Added
- Direct classification: Gaussian Naive Bayes classifier on synthetic 2D data with adjustable class separation and spread; displays ROC curve, AUC, confusion matrix, and classification metrics
- Inverse classification: direct confusion matrix manipulation (TP, FP, TN, FN) with metric calculation, ROC curve generation, and score distribution simulation
- Common utilities: shared JavaScript module with ROC curve computation, confusion matrix rendering, metric calculations, and drag-and-drop functionality
- Base styling: responsive CSS framework with chart cards, floating controls, and unified visual design

[0.2.0]: https://github.com/berangerthomas/SchoolOfStatistics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/berangerthomas/SchoolOfStatistics/releases/tag/v0.1.0
