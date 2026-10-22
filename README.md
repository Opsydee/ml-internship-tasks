# Machine Learning Program Tasks

Solutions to the tasks from a 4-week machine learning program. Each task lives in its own folder with a notebook, its data, and the figures it produces.

| Task | Topic | Folder |
|---|---|---|
| 1 | Multi-class benchmark classification pipeline (Iris) | [task-1-iris-classification](task-1-iris-classification/) |

**Author:** Ojedokun Opeyemi Emmanuel · [GitHub](https://github.com/Opsydee)

---

## Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

### The problem

Build a supervised classifier on the Iris dataset. Scale the features, split the data 80/20, train Logistic Regression and a Decision Tree, then compare them on accuracy, precision, recall and F1-score. Plot a confusion matrix heatmap for each model and a decision boundary visualization.

Notebook: [`task1_iris_classification.ipynb`](task-1-iris-classification/task1_iris_classification.ipynb)

### Data

The classic Iris set: 150 flowers, 4 numeric measurements (sepal length and width, petal length and width), 3 species with 50 samples each. There are no missing values. One row is an exact duplicate of another; I left it in, since it's a known quirk of the dataset and changes nothing at this size.

### Approach

1. Load `iris.csv` with Pandas and check shape, missing values, duplicates and class balance.
2. Draw a Seaborn pairplot colored by species to see which classes overlap before modeling.
3. Split 80/20 with stratification (120 train, 30 test, 10 test samples per species), `random_state=42`.
4. Train Logistic Regression and a Decision Tree. Each model sits in a scikit-learn `Pipeline` with `StandardScaler`, so the scaler learns its mean and standard deviation from training data only.
5. Score both models on the held-out test set, and also with 5-fold stratified cross-validation.
6. Plot confusion matrix heatmaps and decision boundaries.

Two choices matter here.

**Scaling happens after the split.** Standardizing the whole dataset first lets test-set statistics leak into training. Putting the scaler inside the pipeline prevents that. The Decision Tree doesn't need scaling, because tree splits don't depend on feature scale. It's in the pipeline so both models are treated the same way.

**Cross-validation sits next to the test score.** With 30 test samples, one wrong prediction moves accuracy by 3.3 points. The cross-validated score uses all 150 samples and is a steadier number.

### Results

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 | 0.953 ± 0.045 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 | 0.953 ± 0.034 |

Precision, recall and F1 are macro averages (the unweighted mean over the three species). Per class, both models score 1.00 on setosa and 0.90 on versicolor and virginica.

All four test metrics come out identical because the test set is balanced and each model makes one error in each direction. Given those two conditions, they have to be equal. It isn't a computation bug.

**Pairplot**

![Pairplot of the four features colored by species](https://github.com/Opsydee/ml-internship-tasks/blob/main/task1_iris_classification.ipynb/outputs/01_pairplot.png)

**Confusion matrices.** Both models produce the same matrix: all 10 setosa correct, 9 of 10 versicolor, 9 of 10 virginica. The two mistakes are one versicolor predicted as virginica and one virginica predicted as versicolor.

![Confusion matrix heatmaps for both models](https://github.com/Opsydee/ml-internship-tasks/blob/main/task1_iris_classification.ipynb/outputs/02_confusion_matrices.png)

**Decision boundaries.** A boundary can only be drawn in two dimensions, so these plots use petal length and petal width and train a separate two-feature model for each algorithm. They illustrate how each algorithm carves up the space. They don't show the four-feature models from the table above.

![Decision boundaries for both models on petal length and width](https://github.com/Opsydee/ml-internship-tasks/blob/main/task1_iris_classification.ipynb/outputs/03_decision_boundaries.png)

### What I take from this

- Setosa separates cleanly from the others in every view, mostly through petal size.
- Every error is a versicolor/virginica mix-up. The pairplot shows why: those two clusters overlap on all four features.
- The two models can't be told apart here. Test scores tie, and the cross-validated means tie too, with a gap between standard deviations (0.045 versus 0.034) that is small next to the fold-to-fold noise.
- The two boundary plots look different in kind. Logistic Regression draws straight lines. The Decision Tree draws axis-aligned rectangles and, near petal width 1.7, a thin horizontal strip. That strip looks like the tree fitting individual training points rather than a real species boundary, which is the usual way an unpruned tree overfits.

### Limitations

- Iris is small and close to linearly separable. High scores here say little about how either model would do on harder data.
- The headline test numbers come from one split (seed 42). Cross-validation is the better estimate, and I haven't tested other seeds.
- I used default hyperparameters and did no tuning. A depth-limited tree would probably clean up the strip in the boundary plot, but I haven't verified that.
- With 30 test samples, differences of a few points between models aren't meaningful.

### Files

```
task-1-iris-classification/
├── task1_iris_classification.ipynb   # full pipeline, run top to bottom
├── iris.csv                          # dataset
└── outputs/
    ├── 01_pairplot.png
    ├── 02_confusion_matrices.png
    ├── 03_decision_boundaries.png
    └── results.csv                   # metrics table
```

### How to run

```
pip install -r requirements.txt
jupyter lab
pandas
numpy
scikit-learn>=1.1
seaborn
matplotlib
jupyterlab
ipython



Needs scikit-learn 1.1 or newer for `DecisionBoundaryDisplay`.
