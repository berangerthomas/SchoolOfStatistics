# School of Statistics - Interactive Classification Dashboards

![Logo](./src/assets/logo.jpg)

Welcome to the Interactive Classification Dashboards project ! This repository contains a set of tools designed to help users understand the core concepts of binary classification in machine learning through hands-on, visual interaction.

## 🚀 About This Project

### 🌐 Live Demos

*   **[Direct Classification Dashboard](https://berangerthomas.github.io/SchoolOfStatistics/direct_classifier.html)**
*   **[Inverse Classification Dashboard](https://berangerthomas.github.io/SchoolOfStatistics/inverse_classifier.html)**

This project provides two distinct interactive dashboards:

1.  **Direct Classification Dashboard (`direct_classifier.html`)** : This tool allows you to generate a synthetic 2D dataset for two classes. You can adjust the **class separation** and **data spread (standard deviation)** to see how these parameters affect the performance of a Gaussian Naive Bayes classifier. The dashboard visualizes:
    *   The generated data points.
    *   The resulting ROC curve and its Area Under the Curve (AUC).
    *   Key performance metrics (Accuracy, Precision, Recall, etc.).
    *   A detailed confusion matrix.

2.  **Inverse Classification Dashboard (`inverse_classifier.html`)**: This tool works in reverse. Instead of generating data, you directly manipulate the values of the **confusion matrix** (True Positives, False Positives, True Negatives, and False Negatives). The application then simulates a distribution of classifier scores that would lead to your specified matrix and visualizes the resulting metrics, ROC curve, and score distribution. This provides a unique, intuitive way to understand the relationships between the confusion matrix and other performance indicators.

## 📂 Project Structure

The project has been organized into a clean and maintainable structure:

```
.
├── direct_classifier.html
├── inverse_classifier.html
├── LICENSE
├── README.md
└── src
    ├── assets
    │   └── logo.jpg
    ├── css
    │   ├── inverse_style.css
    │   └── style.css
    └── js
        ├── direct_classifier.js
        └── inverse_classifier.js
```

*   **`direct_classifier.html`**: The main page for the direct classification tool.
*   **`inverse_classifier.html`**: The main page for the inverse classification tool.
*   **`src/`**: Contains all source assets.
    *   **`assets/`**: Stores static assets like the project logo.
    *   **`css/`**: Contains the stylesheets for the HTML pages.
    *   **`js/`**: Contains the JavaScript logic for each interactive dashboard.
*   **`LICENSE`**: The project's license file.
*   **`README.md`**: This file.

## 🛠️ How to Use

1.  Clone this repository to your local machine.
2.  Open either `direct_classifier.html` or `inverse_classifier.html` in your web browser.
3.  No local server is needed! All the logic is self-contained in the HTML, CSS, and JavaScript files.

Interact with the sliders and controls on each page to explore the concepts of classification.

## 📄 License

This project is distributed under the terms of the license specified in the `LICENSE` file.
