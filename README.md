# 🫘 Dry Bean Classifier Lab — ML Classification Models & Streamlit Deployment

**Live app:** <STREAMLIT_APP_LINK>
**GitHub repository:** https://github.com/AdvenExplorer08/drybean-classifier

---

## a. Problem statement

Dry beans are graded and priced by variety, but sorting them by eye is slow and inconsistent.
The goal of this project is to **predict the variety of a dry bean (7 classes) from 16 shape
measurements extracted from a photo of the grain**, and to compare five classical machine-learning
classifiers on that task.

The work covers the full workflow:

1. Train and tune 5 classifiers on the same dataset: Logistic Regression, Decision Tree, K-Nearest
   Neighbours, Gaussian Naive Bayes and Support Vector Machine.
2. Evaluate each one on a held-out test set with Accuracy, AUC, Precision, Recall, F1 and MCC.
3. Save the trained models and serve them through an interactive Streamlit app, where a user uploads
   test data, picks a model and sees its metrics, confusion matrix and classification report.
   The app only **loads** the saved models; it never retrains on uploaded data.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | **Dry Bean Dataset** |
| Source | UCI Machine Learning Repository, dataset id 602 — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Original study | Koklu, M. & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507 |
| Instances | 13,611 bean grains (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric features (meets the minimum of 12) |
| Target | `Class` — 7 varieties: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY |
| Task type | Multi-class classification |
| Missing values | None |

**How the data was created:** 13,611 grains of 7 registered dry-bean varieties were photographed with a
high-resolution camera. After segmentation, 12 dimension features and 4 shape-form features were
computed from each grain's outline.

**Features**

| # | Feature | Meaning |
|---|---|---|
| 1 | Area | Number of pixels inside the bean outline |
| 2 | Perimeter | Length of the bean's border |
| 3 | MajorAxisLength | Longest line that can be drawn through the bean |
| 4 | MinorAxisLength | Longest line perpendicular to the major axis |
| 5 | AspectRatio | MajorAxisLength / MinorAxisLength |
| 6 | Eccentricity | Eccentricity of the ellipse with the same moments as the bean |
| 7 | ConvexArea | Pixels in the smallest convex polygon containing the bean |
| 8 | EquivDiameter | Diameter of a circle with the same area as the bean |
| 9 | Extent | Bean area / bounding-box area |
| 10 | Solidity | Area / ConvexArea (convexity) |
| 11 | Roundness | 4π·Area / Perimeter² |
| 12 | Compactness | EquivDiameter / MajorAxisLength |
| 13–16 | ShapeFactor1–4 | Four additional shape descriptors defined in the original paper |

**Class distribution (original 13,611 rows)**

| DERMASON | SIRA | SEKER | HOROZ | CALI | BARBUNYA | BOMBAY |
|---|---|---|---|---|---|---|
| 3,546 | 2,636 | 2,027 | 1,928 | 1,630 | 1,322 | 522 |

The classes are imbalanced (BOMBAY is about 4% of the data), so a **stratified** split is used and
Precision, Recall, F1 and AUC are reported as **support-weighted averages** over the 7 classes.

**Pre-processing**

* Dropped 68 exact duplicate rows so the same bean cannot be in both train and test.
* Stratified 80/20 split with `random_state = 21` → **10,834 training rows, 2,709 test rows**.
  The test rows are saved as `test_data.csv`.
* `StandardScaler` (fitted on training data only) is placed inside a `Pipeline` for the
  scale-sensitive models (Logistic Regression, KNN, SVM), so scaling is saved inside each model file.
* Hyper-parameters were tuned with 5-fold stratified cross-validation on the training set only.

## c. GitHub repository link

https://github.com/AdvenExplorer08/drybean-classifier

```
drybean-classifier/
├── app.py                     # Streamlit app (loads saved models, no retraining)
├── model_training.ipynb       # Training, tuning, evaluation and model saving
├── requirements.txt           # Pinned dependencies for Streamlit Community Cloud
├── README.md
├── test_data.csv              # Held-out test split (16 features + Class), 2,709 rows
├── data/
│   └── Dry_Bean_Dataset.csv   # Full dataset from UCI
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── model_info.json        # Feature list, classes, tuned parameters, test metrics
└── .streamlit/config.toml     # App theme
```

## d. Models used

| # | Model | Configuration (best from 5-fold CV) |
|---|---|---|
| 1 | Logistic Regression | StandardScaler + multinomial LogisticRegression, C = 1 |
| 2 | Decision Tree Classifier | max_depth = 8, min_samples_leaf = 10 (no scaling needed) |
| 3 | K-Nearest Neighbour Classifier | StandardScaler + KNN, k = 9, distance-weighted votes |
| 4 | Naive Bayes Classifier | GaussianNB (Gaussian chosen because all features are continuous) |
| 5 | Support Vector Machine | StandardScaler + SVC, RBF kernel, C = 10, gamma = "scale", probability outputs |

### Comparison table — evaluation metrics on the test set (2,709 beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9228 | 0.9945 | 0.9232 | 0.9228 | 0.9230 | 0.9067 |
| Decision Tree | 0.9140 | 0.9859 | 0.9164 | 0.9140 | 0.9146 | 0.8963 |
| KNN | 0.9247 | 0.9898 | 0.9252 | 0.9247 | 0.9248 | 0.9089 |
| Naive Bayes | 0.7630 | 0.9650 | 0.7632 | 0.7630 | 0.7611 | 0.7140 |
| Support Vector Machine | **0.9350** | **0.9955** | **0.9352** | **0.9350** | **0.9351** | **0.9214** |

*AUC = one-vs-rest ROC AUC (weighted). Precision, Recall and F1 = weighted average. MCC = multi-class
Matthews Correlation Coefficient. Weighted recall equals accuracy by definition, which is why those two
columns match.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong linear baseline: 92.3% accuracy, MCC 0.907 and the second-best AUC (0.9945), so its probability ranking is excellent even though it only draws linear boundaries. Train accuracy (92.5%) is almost equal to test accuracy, so it does not overfit. Most of its errors are between **DERMASON and SIRA** (42 + 41 beans), two varieties whose size and shape ranges overlap, which a linear boundary cannot fully separate. |
| **Decision Tree** | Lowest of the four strong models (91.4% accuracy, MCC 0.896, AUC 0.986). Limiting depth to 8 with at least 10 samples per leaf kept the overfitting small (93.7% train vs 91.4% test). It relies mostly on ShapeFactor1, ShapeFactor3 and MajorAxisLength. Its axis-parallel splits struggle most with DERMASON vs SIRA (62 DERMASON beans predicted as SIRA). Its AUC is lower because a tree gives coarse leaf-level probabilities instead of smooth scores. |
| **KNN** | Second-best model (92.5% accuracy, MCC 0.909). Scaling was essential because Area is in the tens of thousands while ShapeFactor2 is around 0.002. Cross-validation chose k = 9 with distance weighting. It scores 100% on training data only because each training bean is its own nearest neighbour, but test accuracy stays high, so it generalises well. It confuses SIRA↔DERMASON and BARBUNYA→CALI. |
| **Naive Bayes** | Clearly the weakest: 76.3% accuracy and MCC 0.714. The reason is its independence assumption: the features are highly correlated (Area vs ConvexArea r = 1.000, Compactness vs ShapeFactor3 r = 0.999, 21 feature pairs above 0.9), so the same information is counted many times and probabilities become over-confident. BARBUNYA recall is only 47.5% (108 of 265 predicted as CALI). Its AUC is still 0.965, meaning it still ranks the right class fairly high, but its final class decisions are often wrong. |
| **Support Vector Machine** | Best on every metric: 93.5% accuracy, 0.9955 AUC, 0.935 F1 and 0.921 MCC. The RBF kernel draws curved boundaries that separate overlapping varieties better than linear or axis-parallel ones. It has the highest recall for DERMASON (94.4%), CALI (95.4%) and BARBUNYA (92.1%). The small gap between train (93.7%) and test (93.5%) shows it generalises well. SIRA remains the hardest class (89.0% recall). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF, C = 10).** It leads on all six metrics, including MCC (0.921), which is the most reliable single score for an imbalanced 7-class problem. KNN and Logistic Regression follow closely (within about 1.2% accuracy), the Decision Tree is a little lower, and Gaussian Naive Bayes is far behind because the shape features are strongly correlated. All five models catch every BOMBAY bean (100% recall; it is much larger than the other varieties), and **SIRA vs DERMASON is the main source of error**. |

## Streamlit app features

* **CSV upload** of test data (the bundled `test_data.csv` can also be downloaded from the sidebar or used directly)
* **Model selection dropdown** for the 5 trained models
* **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
* **Confusion matrix** heat-map and **classification report**
* **Compare all 5 models** tab with a ranked table and bar chart on the uploaded data
* Prediction table with a correct/incorrect flag, downloadable as CSV
* Input validation (missing columns, unknown labels, files containing only some classes)

## How to run locally

```bash
pip install -r requirements.txt
jupyter notebook model_training.ipynb   # optional: retrain and regenerate model/ and test_data.csv
streamlit run app.py
```
