# Linear Discriminant Analysis (LDA) — Wine Customer Segmentation

A machine learning project that uses **Linear Discriminant Analysis (LDA)** to reduce 13 chemical features of wines down to **2 linear discriminants**, then trains a Logistic Regression model to predict which customer segment a wine belongs to.

## What is LDA?

LDA is a **supervised** dimensionality reduction technique. Like PCA, it creates new axes from combinations of the original features, but it chooses them differently:

- **PCA** looks for the directions of **maximum variance** in the data and ignores the labels.
- **LDA** uses the labels to find the directions that **best separate the classes**, maximising the distance between class means while keeping each class tightly grouped.

With 3 classes, LDA can produce at most 2 discriminants (number of classes − 1).

## Dataset

**File:** `Wine.csv`

- 178 wines, each described by 13 chemical properties
- Target: `Customer_Segment` — 3 segments (59, 71 and 48 wines)

| Features |
|---|
| `Alcohol`, `Malic_Acid`, `Ash`, `Ash_Alcanity`, `Magnesium`, `Total_Phenols`, `Flavanoids`, `Nonflavanoid_Phenols`, `Proanthocyanins`, `Color_Intensity`, `Hue`, `OD280`, `Proline` |

## Workflow

The notebook (`linear_discriminant_analysis.ipynb`) follows these steps:

1. **Import libraries** — NumPy, Matplotlib, Pandas
2. **Load dataset** — 13 features `X`, target `y` (`Customer_Segment`)
3. **Train/test split** — 80% training (142 wines), 20% test (36 wines), `random_state=0`
4. **Feature scaling** — StandardScaler applied to all features
5. **Apply LDA** — reduces the 13 features to 2 linear discriminants; note that `fit_transform` receives `y_train`, because LDA is supervised
6. **Model training** — Logistic Regression on the 2 discriminants
7. **Evaluation** — confusion matrix and accuracy score
8. **Visualisation** — decision regions for the training and test sets, plotted on LD1 vs. LD2

## Requirements

```
numpy
matplotlib
pandas
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib pandas scikit-learn
```

## Usage

Open the notebook in Jupyter or Google Colab and run all cells in order:

```bash
jupyter notebook linear_discriminant_analysis.ipynb
```

## Model Configuration

```python
LDA(n_components = 2)
lda.fit_transform(X_train, y_train)   # uses the labels
LogisticRegression(random_state = 0)
```

## Results

**Class separation captured by each discriminant:**

| Discriminant | Share of between-class separation |
|---|---:|
| LD1 | 73.9% |
| LD2 | 26.1% |

**Confusion matrix (test set, 36 wines):**

| | Predicted 1 | Predicted 2 | Predicted 3 |
|---|---:|---:|---:|
| **Actual 1** | 14 | 0 | 0 |
| **Actual 2** | 0 | 16 | 0 |
| **Actual 3** | 0 | 0 | 6 |

**Accuracy: 100%** (all 36 wines classified correctly)

LDA outperforms PCA on this dataset (100% vs. 97.2%), because it uses the segment labels to pick the directions that separate the segments best. The decision-region plots show the three segments forming tight, well-separated clusters along LD1 and LD2.

## Key Concepts

- **Supervised dimensionality reduction**: uses the target labels to choose the new features.
- **Linear discriminant**: a combination of the original features chosen to maximise separation between classes.
- **Between-class vs. within-class variance**: LDA maximises the spread between class means relative to the spread inside each class.
- **Maximum number of components**: number of classes − 1 (here, 2).
- **PCA vs. LDA**: PCA keeps the most variance; LDA keeps the most class separation.
