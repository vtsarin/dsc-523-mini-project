# COMPLETE CLAUDE CODE PROMPT — Titanic Data Mining Mini Project

> **Copy everything below this line and paste into Claude Code.**

---

## PROJECT OVERVIEW

Build a complete, publication-ready data mining mini project on the Titanic dataset for a graduate course (DSC 523 — Data Mining, IIIT Kottayam). The project has TWO deliverables:

1. **A Jupyter Notebook (.ipynb)** — containing the full Python analysis pipeline (data acquisition, cleaning, EDA, feature engineering, modeling, evaluation, visualization)
2. **A LaTeX conference paper compiled to PDF** — a generic conference-format research paper with all figures embedded

Use the draft outline provided below as a **reference for structure, team roles, and narrative direction**, but **rebuild the technical content from scratch** with correct methodology and realistic results.

---

## CRITICAL TECHNICAL FIXES (from the original draft)

The original draft has these problems that MUST be fixed:

1. **100% accuracy is wrong** — The draft reports both models achieving 100% accuracy with a claim that "100% of females survived and 0% of males survived." This is factually incorrect. The real Titanic training data (~891 records) has ~38% overall survival, ~74% female survival, ~19% male survival. The original team likely used Kaggle's `test.csv` (which has NO `Survived` column) incorrectly, or used a corrupted subset. **FIX**: Use ONLY `train.csv` from Kaggle (891 records), split it 80/20 internally with stratification, and report real accuracy figures (expect ~77-85% depending on model).

2. **Dataset description says 418 records** — That's the size of Kaggle's test set, not the training set. The actual labeled training data has 891 records. **FIX**: Use the 891-record training set.

3. **"Gender Paradox" claim is false** — There is no 100%/0% split. **FIX**: Report actual survival rates by gender from the real data.

4. **No cross-validation** — A single train/test split is weak for a conference paper. **FIX**: Use 10-fold stratified cross-validation for robust evaluation, plus a held-out test set for final reporting.

5. **Shallow analysis** — The draft only uses accuracy. **FIX**: Include precision, recall, F1-score, ROC-AUC, confusion matrices, and statistical significance testing between models.

---

## DELIVERABLE 1: JUPYTER NOTEBOOK (`titanic_analysis.ipynb`)

### Structure the notebook with these clearly marked sections:

```
## 1. Project Setup & Data Acquisition
## 2. Data Profiling & Initial Exploration  
## 3. Data Cleaning & Preprocessing
## 4. Exploratory Data Analysis (EDA) & Visualization
## 5. Feature Engineering
## 6. Model Building & Training
## 7. Model Evaluation & Comparison
## 8. Results Summary & Key Findings
```

### Detailed Requirements per Section:

#### 1. Project Setup & Data Acquisition
- Import all libraries at the top (pandas, numpy, matplotlib, seaborn, sklearn, scipy)
- Load the Titanic dataset. Use this approach:
  ```python
  # Option A: Download from URL
  url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
  df = pd.read_csv(url)
  
  # Option B: If no network, use sklearn
  # from sklearn.datasets import fetch_openml
  # titanic = fetch_openml('titanic', version=1, as_frame=True)
  ```
- If neither URL works due to network restrictions, include a fallback that loads from a local file path and add a markdown cell instructing the user to download `train.csv` from Kaggle
- Print shape, dtypes, first 5 rows, .info(), .describe()

#### 2. Data Profiling & Initial Exploration
- Check missing values (count and percentage per column)
- Check class distribution of target variable (Survived)
- Identify variable types: Nominal (Sex, Embarked, Cabin), Ordinal (Pclass), Numerical (Age, Fare, SibSp, Parch)
- Statistical summary for numerical features
- Value counts for categorical features
- Note any anomalies or interesting observations in markdown cells

#### 3. Data Cleaning & Preprocessing
- **Age**: Impute missing values using **median** (justify: Age is right-skewed, median is robust to outliers). Also try group-wise median (by Pclass + Sex) as an improvement — use whichever is better justified.
- **Embarked**: Impute with **mode** (only 2 missing values)
- **Fare**: Check for missing, impute with median of corresponding Pclass if needed
- **Cabin**: Drop (too many missing — ~77% missing). BUT before dropping, extract the deck letter as a new feature `Deck` if possible
- **Drop columns**: PassengerId, Name, Ticket (but extract Title from Name first — see Feature Engineering)
- **Encode categoricals**: 
  - Sex → binary (0/1)
  - Embarked → one-hot encoding (preferred over label encoding for nominal variables)
- Print cleaned dataframe shape and confirm zero missing values

#### 4. EDA & Visualization (minimum 8 plots)
Generate ALL of these visualizations with professional styling:

1. **Survival count plot** — bar chart of Survived (0 vs 1) showing class imbalance
2. **Survival by Gender** — grouped bar chart showing survival rate by Sex
3. **Survival by Pclass** — grouped bar chart showing survival rate by passenger class
4. **Survival by Pclass × Gender** — faceted or grouped plot showing interaction effect
5. **Age distribution** — histogram/KDE, colored by survival status (overlaid)
6. **Fare distribution** — histogram/KDE by survival, possibly log-scaled
7. **Correlation heatmap** — Pearson correlation matrix of all numerical features with annotations
8. **Survival by Embarked port** — bar chart
9. **Age vs Fare scatter** — colored by survival, sized by Pclass
10. **Family size vs survival** — create FamilySize = SibSp + Parch + 1, plot survival rate

Styling requirements:
- Use `plt.style.use('seaborn-v0_8-whitegrid')` or similar clean style
- All plots must have: title, axis labels, legend where applicable
- Use a consistent color palette (e.g., `palette=['#e74c3c', '#2ecc71']` for died/survived)
- Save each plot as a high-resolution PNG (300 dpi) in a `figures/` directory for LaTeX embedding
- Use `plt.tight_layout()` on every figure
- Figure size: (10, 6) for standard plots, (12, 8) for heatmaps

#### 5. Feature Engineering
Create these derived features:
- **Title**: Extract from Name column (Mr, Mrs, Miss, Master, etc.), group rare titles into 'Rare'
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: 1 if FamilySize == 1, else 0
- **AgeBin**: Discretize Age into bins (Child: 0-12, Teenager: 13-19, Adult: 20-40, Middle: 41-60, Senior: 60+)
- **FareBin**: Quartile-based binning of Fare
- **Deck**: First letter of Cabin (before dropping Cabin), with 'U' for unknown

After feature engineering, select final feature set and justify the selection.

#### 6. Model Building & Training

**Primary models (MUST include — matches paper title):**
1. **Gaussian Naive Bayes** (`GaussianNB`)
2. **Decision Tree Classifier** (`DecisionTreeClassifier` with max_depth=3,4,5 — try multiple and pick best via CV)

**Additional models (for richer comparison — Claude's recommendation):**
3. **Logistic Regression** (as a strong baseline)
4. **Random Forest** (ensemble extension of Decision Tree — shows value of bagging)

**Training protocol:**
- Split data: 80% train / 20% test (stratified, random_state=42)
- Use **10-fold Stratified Cross-Validation** on the training set for model selection
- Train final models on full training set, evaluate on held-out test set
- For Decision Tree: also perform **cost-complexity pruning** (ccp_alpha) and show the pruning curve

#### 7. Model Evaluation & Comparison

For EACH model, compute and display:
- **Accuracy** (with 95% confidence interval from CV)
- **Precision, Recall, F1-Score** (per class and weighted average)
- **Confusion Matrix** — plotted as heatmap using `sns.heatmap` or `ConfusionMatrixDisplay`
- **ROC Curve** — all models on the same plot with AUC values in legend
- **Classification Report** — printed in formatted table

Create a **comparison summary table**:
```
| Model              | CV Accuracy (mean±std) | Test Accuracy | Precision | Recall | F1    | AUC   |
|--------------------|------------------------|---------------|-----------|--------|-------|-------|
| Naive Bayes        | ...                    | ...           | ...       | ...    | ...   | ...   |
| Decision Tree      | ...                    | ...           | ...       | ...    | ...   | ...   |
| Logistic Regression| ...                    | ...           | ...       | ...    | ...   | ...   |
| Random Forest      | ...                    | ...           | ...       | ...    | ...   | ...   |
```

Also include:
- **Feature importance plot** from Decision Tree and Random Forest
- **Decision Tree visualization** using `plot_tree()` with max_depth=3 (save as high-res PNG)
- **McNemar's test** or similar statistical test to check if accuracy differences between NB and DT are statistically significant

#### 8. Results Summary
- Markdown cell summarizing key findings
- Which model performed best and why
- Which features were most predictive
- Any surprising discoveries

---

## DELIVERABLE 2: LaTeX CONFERENCE PAPER (`titanic_paper.tex` → `titanic_paper.pdf`)

### Paper Metadata
```
Title: A Comparative Framework for Survival Analysis on the RMS Titanic: 
       Evaluation of Naïve Bayesian and Decision Tree Classifiers

Authors:
1. Sri. Sarin Vadakke Thayyil (2026EMAI10051) — Project Manager & Lead Data Architect
2. Smt. Lekshmi S S (2026EMAI10037) — Principal Researcher & Algorithm Specialist
3. Sri. Akbar TE (2026EMAI10005) — Visualisation Engineer & Data Storyteller
4. Smt. Suja V (2026EMAI10058) — Senior Data Preprocessing Engineer
5. Sri. Nikhil Ravi (2026EMAI10042) — Model Performance Evaluator & Reviewer

Affiliation: IIIT Kottayam, eMTech 2026 Batch
Course: DSC 523 — Data Mining
Instructor: Dr. Anu Maria Sebastian
Date: 25th April 2026
```

### Paper Structure & Content Requirements

Use the `article` documentclass with two-column format. Include these packages: `geometry, graphicx, booktabs, amsmath, amssymb, hyperref, float, caption, subcaption, natbib, algorithm2e or algorithmic, xcolor, listings`.

Page setup: A4, 1-inch margins, 10pt font.

#### Section 1: Abstract (~200 words)
- State the problem (Titanic survival prediction as a benchmark for classification)
- Mention the methods (Naive Bayes, Decision Tree, with Logistic Regression and Random Forest as supplementary)
- Key results (actual accuracy figures from the analysis — do NOT fabricate, use placeholder `XX.X\%` if results aren't known yet, but ideally run the notebook first and paste real numbers)
- Significance (comparative framework for probabilistic vs. rule-based classification)

#### Section 2: Introduction (~400-500 words)
- Historical context of the Titanic disaster (1912, 1514 deaths, "women and children first")
- Why this dataset is a benchmark in data mining education
- Frame within KDD process (Selection → Preprocessing → Transformation → Mining → Interpretation)
- State research objectives clearly (numbered list):
  1. Apply systematic data preprocessing and feature engineering
  2. Compare probabilistic (NB) vs. rule-based (DT) classification approaches
  3. Evaluate which factors most strongly predict survival
  4. Benchmark against ensemble methods (LR, RF) for context
- Paper organization paragraph

#### Section 3: Literature Review (~500-600 words)
Cover these topics with proper citations:

**3.1 Bayesian Classification**
- Bayes' theorem: $P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$
- Class conditional independence assumption
- Strengths: computationally efficient, handles missing data well, works with small datasets
- Limitations: independence assumption rarely holds, poor with correlated features
- Cite: Han, Kamber & Pei (2011); Mitchell (1997)

**3.2 Decision Tree Induction**
- Gini Index: $Gini(D) = 1 - \sum_{i=1}^{m} p_i^2$
- Information Gain / Entropy: $Entropy(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)$
- Greedy top-down recursive partitioning
- Pruning strategies (pre-pruning via max_depth, post-pruning via cost-complexity)
- Cite: Quinlan (1986); Breiman et al. (1984)

**3.3 Related Work on Titanic Dataset**
- Mention 2-3 prior studies or Kaggle approaches (general framing is fine)
- Note common accuracy ranges reported in literature (typically 78-85%)

#### Section 4: Dataset Description (~300-400 words)
- Source: Kaggle Titanic competition (train.csv, 891 records, 12 features)
- Feature table (LaTeX `tabular` with booktabs):
  | Feature | Type | Missing | Description |
- Class distribution: ~38% survived, ~62% died
- Key statistics from profiling
- Include 1 figure: class distribution bar chart

#### Section 5: Methodology (~600-800 words)

**5.1 Data Preprocessing**
- Missing value treatment (Age: median imputation; Embarked: mode; Cabin: dropped after deck extraction)
- Feature selection rationale
- Encoding strategy

**5.2 Feature Engineering**
- Title extraction, FamilySize, IsAlone, binning strategies
- Justify each derived feature with domain reasoning

**5.3 Classification Models**
- Formal description of each model (NB, DT, LR, RF)
- Hyperparameters used and justification
- Include pseudocode or algorithm block for at least NB or DT

**5.4 Evaluation Protocol**
- 80/20 stratified split
- 10-fold stratified cross-validation
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Statistical significance testing

Include a **methodology flowchart** as a figure (generate as a clean diagram — either TikZ in LaTeX or save from Python as PNG).

#### Section 6: Results & Discussion (~600-800 words)

**6.1 EDA Findings**
- Key visualizations (embed 4-5 of the best figures from the notebook)
- Discuss survival patterns: gender, class, age, fare, family size
- Correlation analysis insights

**6.2 Model Performance**
- Performance comparison table (LaTeX `tabular` with booktabs)
- Confusion matrices (as figure with subfigures for NB and DT side by side)
- ROC curves (single figure, all models overlaid)
- Feature importance (from DT/RF)
- Decision tree visualization

**6.3 Discussion**
- Why DT likely outperforms NB on this dataset (correlated features violate independence assumption)
- Significance of gender and class as predictors — historical context
- Comparison with published benchmarks
- Limitations of the study

#### Section 7: Implications & Applications (~200-300 words)
- Risk assessment and insurance (Bayesian updating)
- Emergency response and crisis planning
- Algorithmic fairness and bias auditing (gender/class bias in predictions)
- Educational value for data mining pedagogy

#### Section 8: Individual Contributions
Use a LaTeX table:
```
| Member | Role | Contributions |
```
With these assignments:
- **Sarin Vadakke Thayyil**: Project Manager & Lead Data Architect — directed group, defined train-test strategy, authored Methodology and Implications
- **Lekshmi S S**: Principal Researcher & Algorithm Specialist — comparative analysis of Gini vs Entropy, drafted Literature Review
- **Akbar TE**: Visualisation Engineer & Data Storyteller — all visualizations, data storytelling, professional visual layout
- **Suja V**: Senior Data Preprocessing Engineer — data cleaning, median imputation, redundant attribute identification
- **Nikhil Ravi**: Model Performance Evaluator & Reviewer — confusion matrix verification, model evaluation, ensured compliance with course guidelines

#### Section 9: Conclusion & Future Work (~200-300 words)
- Summarize key findings
- Best performing model and why
- Future work: deep learning (MLP), ensemble stacking, SHAP explanations, hyperparameter optimization with GridSearchCV

#### References (minimum 8-10)
Use these references (add more as appropriate):
1. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*. 3rd ed. Morgan Kaufmann.
2. Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81-106.
3. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.
4. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
5. Kaggle. (2012). Titanic: Machine Learning from Disaster. https://www.kaggle.com/c/titanic
6. Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques*. 3rd ed. Morgan Kaufmann.
7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. 2nd ed. Springer.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
9. IIIT Kottayam Courseware (2026). DSC 523: Data Mining.

---

## FILE STRUCTURE

Create this directory structure:
```
titanic_project/
├── titanic_analysis.ipynb          # Complete Jupyter Notebook
├── titanic_paper.tex               # LaTeX source
├── titanic_paper.pdf               # Compiled PDF (if pdflatex available)
├── figures/                        # All generated plots
│   ├── survival_countplot.png
│   ├── survival_by_gender.png
│   ├── survival_by_class.png
│   ├── survival_by_class_gender.png
│   ├── age_distribution.png
│   ├── fare_distribution.png
│   ├── correlation_heatmap.png
│   ├── survival_by_embarked.png
│   ├── age_fare_scatter.png
│   ├── family_size_survival.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_dt.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── decision_tree_viz.png
│   └── methodology_flowchart.png
├── data/
│   └── titanic.csv                 # Downloaded dataset
└── README.md                       # Project overview
```

---

## EXECUTION ORDER

1. **First**: Create the directory structure
2. **Second**: Download the Titanic dataset (try the GitHub URL, fallback to sklearn)
3. **Third**: Build and execute the Jupyter Notebook — run ALL cells and save outputs
4. **Fourth**: Ensure all figures are saved to `figures/` directory
5. **Fifth**: Build the LaTeX paper, embedding the actual figures and real results from the notebook
6. **Sixth**: Compile LaTeX to PDF using `pdflatex` (run twice for references)
7. **Seventh**: Create README.md

---

## QUALITY STANDARDS

- **Academic tone**: Write like a graduate-level conference paper, not a blog post
- **No fabricated results**: Every number in the paper must come from actual code execution
- **Professional visualizations**: Publication-quality figures, no default matplotlib styling
- **Proper LaTeX**: Clean typesetting, proper math environments, no compilation errors
- **Reproducible**: Anyone should be able to run the notebook and get the same results (use random_state=42 everywhere)
- **Code quality**: Well-commented, logically organized, PEP 8 compliant
- **The paper should read as an independent document** — someone should understand the full study without seeing the notebook

---

## IMPORTANT NOTES

- This is for DSC 523 (Data Mining) at IIIT Kottayam, taught by Dr. Anu Maria Sebastian
- The course covers: KDD process, data preprocessing, classification, association rules, clustering, neural networks, data visualization
- Frame the methodology within the KDD process where appropriate
- The team has 5 members — acknowledge all contributions in the paper
- The primary comparison is Naive Bayes vs Decision Tree (as per the title), with LR and RF providing additional context
- DO NOT report 100% accuracy. Real Titanic models typically achieve 77-85%. If your results show >90%, something is likely wrong — check for data leakage.