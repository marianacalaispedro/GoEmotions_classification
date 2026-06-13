# Multi-Label Emotion Classification on GoEmotions

A comprehensive NLP pipeline for **multi-label emotion classification** of Reddit comments, built on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset. The project covers the full machine-learning lifecycle — from data exploration and text preprocessing through baseline evaluation, fine-tuning, and error analysis.

> **Course:** Artificial Intelligence Lab — Sapienza Università di Roma

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Data Exploration](#1-data-exploration)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Baseline Evaluation](#3-baseline-evaluation)
  - [4. Fine-Tuning](#4-fine-tuning)
  - [5. Attention Visualization](#5-attention-visualization)
  - [6. Sentiment-Aware Correction](#6-sentiment-aware-correction)
  - [7. Error Analysis](#7-error-analysis)
- [Models Compared](#models-compared)
- [Key Results](#key-results)
- [Setup & Requirements](#setup--requirements)
- [Usage](#usage)

---

## Overview

Emotion detection in text is inherently a **multi-label** problem — a single sentence can express *joy* and *gratitude* simultaneously, or *anger* alongside *disappointment*. This project:

1. Explores and preprocesses the GoEmotions corpus (58k Reddit comments, 28 emotion labels).
2. Evaluates several **zero-shot baselines** (BERT-based and RoBERTa-based models) under three preprocessing intensities (minimal, medium, harsh).
3. Evaluates an **Ekman emotion mapping** (6 coarse-grained emotions) as an alternative label scheme.
4. **Fine-tunes BERT** (`bert-base-uncased`) on the GoEmotions training split for direct multi-label classification.
5. Applies **VADER sentiment analysis** as a post-prediction correction heuristic.
6. Provides **attention visualizations** to interpret model predictions.
7. Conducts a thorough **error analysis** across models, examining error patterns by emotion, label cardinality, and text length.

---

## Project Structure

```
.
├── data/
│   ├── GoEmotions.csv                        # Raw GoEmotions dataset
│   └── goemotions_multilabel.csv             # Preprocessed multi-label version
│
├── notebooks/
│   ├── exploration.ipynb                     # EDA: label distributions, text stats, word clouds, co-occurrence
│   ├── preprocessing.ipynb                   # Text cleaning pipeline (minimal / medium / harsh)
│   ├── baseline_evaluation.ipynb             # Zero-shot model evaluation + fine-tuning + error analysis
│   ├── attention_visualization.ipynb         # Attention head visualizations for BERT
│   └── wordclouds/                           # Saved word cloud images
│
├── source/
│   ├── general_preprocessing.py              # MainPipeline class: regex cleaning, tokenization, lemmatization,
│   │                                         #   vectorization (TF-IDF, BoW, Doc2Vec), translation, NER features,
│   │                                         #   and spelling correction utilities
│   ├── modeling.py                           # Model loading, prediction, multi-label metrics, dataset classes
│   │                                         #   for fine-tuning (GoEmotionsDataset), and Trainer metrics
│   ├── visualizations.py                     # Plotly & Matplotlib chart helpers: bar, histogram, heatmap,
│   │                                         #   donut, treemap, word clouds, co-occurrence networks
│   ├── error_analysis.py                     # Error analysis: per-emotion errors, error combinations,
│   │                                         #   cardinality analysis, text-length analysis, side-by-side comparison
│   ├── sentiment_analysis.py                 # VADER sentiment features & sentiment-aware prediction correction
│   └── my_utils.py                           # Dataset I/O utilities (CSV / Excel load & export)
│
├── model_comparison_complete.csv             # Metrics for all models (original test set)
├── model_comparison_with_finetuned.csv       # Metrics including the fine-tuned BERT model
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| **Name** | GoEmotions |
| **Source** | Google Research — Reddit comments |
| **Size** | ~58,000 comments |
| **Labels** | 28 emotions (multi-label) |
| **Label scheme** | Binary indicators per emotion |
| **Alternative grouping** | Ekman 6 (anger, disgust, fear, joy, sadness, surprise) + neutral |

---

## Pipeline

### 1. Data Exploration

- Label frequency distributions and co-occurrence analysis
- Text length statistics (word count, character count)
- Word clouds per preprocessing level and POS tag
- Co-occurrence heatmaps and network graphs

### 2. Preprocessing

Three preprocessing intensities are applied via `MainPipeline`:

| Level | Description |
|---|---|
| **Minimal** | Lowercasing, emoji removal, URL removal, hashtag stripping |
| **Medium** | Minimal + stopword removal, lemmatization, punctuation removal |
| **Harsh** | Medium + repeated-character normalization, diacritic conversion, spelling correction |

Additional capabilities include language detection and translation (via `langdetect`, `langid`, and `deep-translator`) and NER feature extraction with BIO alignment.

### 3. Baseline Evaluation

Zero-shot inference using Hugging Face `text-classification` pipelines:

- **`SamLowe/roberta-base-go_emotions`** (RoBERTa-based)
- **`bert-base-uncased` variants** (BERT-based)
- **Ekman mapping** (6-emotion grouping)

Each model is evaluated under all three preprocessing levels.

### 4. Fine-Tuning

- **Model:** `bert-base-uncased` fine-tuned for multi-label classification
- **Loss:** Binary cross-entropy (per-label sigmoid)
- **Training:** Hugging Face `Trainer` API with custom metrics callback
- **Dataset class:** `GoEmotionsDataset` (tokenization + binary label tensors)

### 5. Attention Visualization

Attention head heatmaps are generated to interpret which tokens the fine-tuned model attends to when predicting specific emotions.

### 6. Sentiment-Aware Correction

A VADER-based post-processing heuristic:
- If VADER detects strong positive sentiment but the model predicts mostly negative emotions, negative predictions are suppressed (and vice versa).
- Controlled by a compound-score threshold.

### 7. Error Analysis

Comprehensive error analysis across all models:
- **Per-emotion:** TP, FP, FN counts and error rates
- **Error combinations:** Most frequent TP/FP/FN patterns per sentence
- **Cardinality analysis:** Error rates vs. number of true emotions per sentence
- **Text-length analysis:** Error rates across text-length bins
- **Side-by-side model comparison** with exported Excel reports

---

## Models Compared

| Model | Preprocessing | Micro F1 | Macro F1 | Samples F1 |
|---|---|---|---|---|
| **Fine-tuned BERT** | Text Only | **0.587** | **0.492** | **0.629** |
| BERT | Minimal | 0.482 | 0.386 | 0.550 |
| BERT | Medium | 0.463 | 0.361 | 0.528 |
| RoBERTa | Minimal | 0.430 | 0.325 | 0.480 |
| RoBERTa | Medium | 0.417 | 0.309 | 0.464 |
| BERT | Harsh | 0.411 | 0.264 | 0.471 |
| RoBERTa | Harsh | 0.388 | 0.247 | 0.437 |
| Ekman | Medium | 0.100 | 0.056 | 0.093 |
| Ekman | Minimal | 0.099 | 0.054 | 0.091 |
| Ekman | Harsh | 0.094 | 0.053 | 0.088 |

> The **fine-tuned BERT** model achieves the best performance across all metrics. Heavier preprocessing degrades performance for pretrained transformer models, as they benefit from the original token distributions they were trained on.

---

## Setup & Requirements

### Prerequisites

- Python 3.9+
- A Hugging Face account and API token (for fine-tuned model access)

### Installation

```bash
pip install pandas numpy scikit-learn torch transformers datasets
pip install nltk emoji unidecode tqdm
pip install plotly matplotlib seaborn wordcloud networkx
pip install vaderSentiment langdetect langid deep-translator
pip install python-Levenshtein jellyfish gensim python-dotenv
pip install huggingface_hub openpyxl
```

### NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Hugging Face Token

Create a `.env` file in the project root:

```
HUGGINGFACE_TOKEN=your_token_here
```

---

## Usage

1. **Exploration** — Open and run `notebooks/exploration.ipynb` to visualize the dataset.
2. **Preprocessing** — Run `notebooks/preprocessing.ipynb` to generate the three preprocessing levels.
3. **Evaluation & Fine-Tuning** — Run `notebooks/baseline_evaluation.ipynb` for zero-shot baselines, fine-tuning, sentiment correction, and error analysis.
4. **Attention Visualization** — Run `notebooks/attention_visualization.ipynb` to inspect attention patterns.

All reusable logic lives in the `source/` modules and is imported by the notebooks.