# CLI Face Recognition

Command line tool for face recognition using some descriptors (algorithm or deep learning), and some metrics to evaluate the results.

## About

This project was developed as second project of the discipline of Digital Image Processing of the Computer Science course at the State University of São Paulo (UNESP) - Bauru

### Description

Minimum requirements:

- 2 descriptors (LBP and any other using deep features)
- 1 database (ARFACE or FRGC)
- Comparison between descriptors
- Metrics: ROC curves, EER, FAR, FRR, F-Score, CMC curves and AUC (Area Under Curve)

Performance evaluation: Authentication (1:1) and Identification (1:n)

### How to run

1. Clone this repository
2. Run `pipenv install` to install dependencies
3. Run `pipenv shell` to activate virtual environment

## Usage

Use the following command to check the usage of the tool:

```bash
python -m face_recognition --help
```
