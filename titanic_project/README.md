# Titanic Survival Analysis — DSC 523 Data Mining Mini Project

**Institution:** Indian Institute of Information Technology, Kottayam
**Course:** DSC 523 — Data Mining · eMTech (AI) 2026 Batch
**Instructor:** Dr. Anu Maria Sebastian
**Date:** 25 April 2026

A reproducible end-to-end KDD pipeline that compares Gaussian Naïve Bayes with a
Decision Tree classifier on the Kaggle Titanic dataset, with Logistic Regression
and Random Forest as supplementary baselines.

## Team

| Member | Roll No | Role |
|---|---|---|
| Sarin Vadakke Thayyil | 2026EMAI10051 | Project Manager & Lead Data Architect |
| Lekshmi S S | 2026EMAI10037 | Principal Researcher & Algorithm Specialist |
| Akbar T E | 2026EMAI10005 | Visualisation Engineer & Data Storyteller |
| Suja V | 2026EMAI10058 | Senior Data Preprocessing Engineer |
| Nikhil Ravi | 2026EMAI10042 | Model Performance Evaluator & Reviewer |

## Key Findings

| Model | CV Accuracy | Test Accuracy | AUC |
|---|---|---|---|
| Naïve Bayes | 0.6572 ± 0.1169 | 0.6257 | 0.7782 |
| **Decision Tree** (depth 4) | **0.8202 ± 0.0363** | **0.8156** | **0.8375** |
| Logistic Regression | 0.8146 ± 0.0393 | 0.8324 | 0.8644 |
| Random Forest | 0.8131 ± 0.0336 | 0.8101 | 0.8286 |

- Decision Tree beats Naïve Bayes by ~19 percentage points (McNemar exact test: p = 8.22×10⁻⁶).
- Top predictors: `Title_Mr`, `Pclass`, `FamilySize`, `Fare` — consistent with the "women and children first" protocol.
- Linear structure is nearly sufficient after feature engineering (Logistic Regression wins on AUC).

## Files

```
titanic_project/
├── titanic_analysis.ipynb     Executed Jupyter notebook (full pipeline, all outputs)
├── titanic_paper.tex          LaTeX source
├── titanic_paper.pdf          Compiled 7-page conference-style paper
├── run_analysis.py            Standalone script that regenerates figures + results.json
├── build_notebook.py          Builds + executes the notebook programmatically
├── make_flowchart.py          Produces the KDD methodology flowchart
├── results.json               Every number quoted in the paper (for verification)
├── data/titanic.csv           Kaggle train.csv (891 rows)
└── figures/                   All 17 PNG figures at 300 dpi
```

## Reproducing the results

```bash
# One-time setup
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter nbformat

# Regenerate all figures and results.json
python run_analysis.py

# Build and execute the notebook
python build_notebook.py

# Compile the paper (requires pdflatex with listings, algorithm2e, authblk)
pdflatex titanic_paper.tex
pdflatex titanic_paper.tex   # second pass resolves cross-references
```

All stochastic components use `random_state = 42`, so numbers are bit-for-bit reproducible.

## Dataset

Kaggle `train.csv` (891 labelled rows, 12 attributes). The Kaggle `test.csv` is
unlabelled and is deliberately *not* used — reports of 100% accuracy in the
literature nearly always stem from confusing these two files.

## License

Academic coursework. Not for redistribution outside IIIT Kottayam DSC 523.
