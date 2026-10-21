# Phishing Website Detection

A machine learning project for identifying potentially phishing URLs through URL-based feature engineering and classification.

## Overview

This project explores how machine learning techniques can be applied to phishing URL detection. It combines URL structure analysis, text-based feature extraction, and classification models to distinguish legitimate URLs from potentially malicious ones.

## Key Features

- **URL Feature Engineering:** Extract URL length, domain characteristics, subdomain information, special-character counts, HTTPS indicators, and path/query features.
- **Text Feature Extraction:** Explore TF-IDF and character-level n-gram representations of URLs.
- **Machine Learning:** Implement a Random Forest classifier and explore an LSTM-based sequence model.
- **Model Evaluation:** Use a train/test split and classification metrics to evaluate model performance.
- **Dataset Processing:** Process labeled URL data for machine learning experiments.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- URL parsing and text feature extraction

## Project Structure

```text
phishing-website-detection/
├── README.md
├── main.py
├── requirements.txt
├── data/
├── results/
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/phishing-website-detection.git
cd phishing-website-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Prepare a labeled CSV file containing URL samples and their corresponding phishing labels. Update the data path and column names in the code if necessary.

### 4. Run the project

```bash
python main.py
```

## Methodology

The project investigates multiple sources of URL information:

1. Structural features, including URL length, domain length, subdomains, and special characters.
2. Text representations using TF-IDF and character-level n-grams.
3. Classification using Random Forest and an LSTM sequence model.
4. Evaluation using classification metrics.

## Limitations and Future Improvements

- Validate the dataset quality, label distribution, and source provenance.
- Prevent data leakage by fitting feature transformers on training data only.
- Evaluate models on held-out data and report precision, recall, F1-score, and confusion matrices.
- Test generalization on independently collected, real-world phishing and legitimate URLs.
- Improve reproducibility through dependency pinning, configuration management, and automated tests.

## Disclaimer

This project is intended for educational and defensive cybersecurity research. URL labels and model predictions should not be treated as definitive security judgments.
