# 🎓 Student Performance Prediction — End-to-End ML Pipeline

Modular machine learning pipeline for predicting student math scores based on demographic and academic preparation features, with proper data ingestion, transformation, and model training components.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient_Boosting-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Predict student math scores based on demographic information and test preparation status. The dataset is the [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) from Kaggle.

## Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| gender | Categorical | Student's gender |
| race_ethnicity | Categorical | Ethnic group (A–E) |
| parental_level_of_education | Categorical | Highest parental education level |
| lunch | Categorical | Standard or free/reduced lunch program |
| test_preparation_course | Categorical | Whether the student completed test prep |
| reading_score | Numerical | Score on reading exam |
| writing_score | Numerical | Score on writing exam |
| **math_score** | **Target** | **Score on math exam** |

## Architecture

```
src/
├── components/
│   ├── data_ingestion.py       # Load data, train/test split, save artifacts
│   ├── data_transformation.py  # ColumnTransformer pipeline (impute + scale + encode)
│   └── model_trainer.py        # Model training and evaluation
├── pipeline/
│   ├── train_pipeline.py       # End-to-end training orchestration
│   └── predict_pipeline.py     # Inference pipeline
├── exception.py                # Custom exception with file/line tracing
├── logger.py                   # Timestamped file logging
└── utils.py                    # Object serialization utilities
```

## Preprocessing Pipeline

| Step | Numerical Features | Categorical Features |
|------|--------------------|---------------------|
| Imputation | Median | Most frequent |
| Encoding | — | One-Hot Encoding |
| Scaling | StandardScaler | StandardScaler |

Built with `sklearn.pipeline.Pipeline` and `ColumnTransformer` for leak-free preprocessing.

## Engineering Highlights

- **@dataclass configuration** — Clean, type-safe path management
- **Custom exception handling** — Captures filename and line number for debugging
- **Structured logging** — Timestamped logs to `logs/` directory
- **Serialization with dill** — Robust object persistence for preprocessor and model artifacts
- **Installable package** — `setup.py` enables `pip install -e .`

## Notebooks

| Notebook | Contents |
|----------|----------|
| `1. EDA Student Performance` | Exploratory data analysis — distributions, correlations, group comparisons |
| `2. Model Training` | CatBoost model training and evaluation |

## Getting Started

```bash
git clone https://github.com/eboekenh/mlproject.git
cd mlproject
pip install -r requirements.txt
```

### Run Data Pipeline
```bash
python -m src.components.data_ingestion
```

## Tech Stack

- **Python 3.8+** — Core language
- **Scikit-learn** — Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder
- **CatBoost** — Gradient boosting model
- **Pandas / NumPy** — Data manipulation
- **dill** — Object serialization

## License

MIT
