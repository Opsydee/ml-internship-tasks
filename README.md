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




### Task 2: California Housing Price Prediction (OLS, Ridge, Lasso)
The problem

Predict median house values for California districts with Ordinary Least Squares, Ridge (L2) and Lasso (L1) regression. Impute missing values, engineer interaction features, deal with multicollinearity, tune Ridge and Lasso with GridSearchCV, and compare the three models on RMSE, MAE and R².

Notebook: task2_california_housing.ipynb

Data

20,640 districts with 8 numeric features, one categorical feature (ocean_proximity, 5 values) and the target median_house_value. total_bedrooms has 207 missing values. There are no duplicate rows. Two details affect the results:

965 districts have a target value of exactly 500,001. That is a cap in the data, not a real price.
The ISLAND category has only 5 rows (4 in training, 1 in the test set).
Approach
Split 80/20 (16,512 training rows, 4,128 test rows, random_state=42) before any preprocessing.
Engineer features: rooms per household, bedrooms per room, population per household, and two interactions that multiply median_income by rooms per household and by housing age.
Build a pipeline that fills missing values with the median, standardizes numeric columns and one-hot encodes ocean_proximity (first category dropped). Everything is fit on training data only.
Inspect a correlation heatmap and variance inflation factors (VIF) for multicollinearity.
Tune Ridge and Lasso with GridSearchCV (5-fold, RMSE scoring, 29 alpha values from 0.001 to 10,000).
Compare OLS, Ridge and Lasso on the held-out test set and plot predicted vs actual values and residuals.

Multicollinearity. The raw count columns are heavily collinear (VIF about 51 for total_bedrooms, 47 for households, 20 for total_rooms). I tried dropping them. That made predictions worse, so I kept them and let Ridge and Lasso handle the collinearity.

OLS feature set	CV RMSE	Test R²
Original 8 features	68,604	0.6254
Original + engineered (used)	67,675	0.6383
Collinear raw counts dropped	70,324	0.6092
Results
Model	Best alpha	CV RMSE	Test RMSE	Test MAE	Test R²
OLS	n/a	67,675	68,845	49,459	0.6383
Ridge (L2)	31.6	67,659	68,935	49,437	0.6374
Lasso (L1)	100	67,661	68,937	49,427	0.6373

Correlation heatmap

Correlation heatmap of the features and target

Cross-validated RMSE across alpha. The Ridge and Lasso curves flatten out near the OLS line, which is why tuning changes so little.

Cross-validated RMSE across the alpha grid

Predicted vs actual

Predicted vs actual values for OLS, Ridge and Lasso

Residual distributions

Residual distributions and residuals vs predicted for OLS

What I take from this
The three models perform the same. Cross-validated RMSE differs by about $15 on a base near $67,700, and OLS is slightly ahead on the test set. With 16,512 training rows and 17 features, OLS has little variance for a penalty to remove. I did not measure fold-to-fold variation, so I can't put a formal bound on that.
Regularization does change one thing. The ISLAND dummy, with 4 training rows, gets a coefficient of about 131,000 under OLS (standardized units), about 15,000 under Ridge and zero under Lasso. It was the only coefficient Lasso removed (1 of 17).
Feature engineering helped a little: it lowered OLS cross-validated RMSE by about 1.4%.
With strongly correlated features, individual coefficients are unreliable. For example, households comes out positive while population comes out negative. I would not read those signs causally.
Limitations
R² is about 0.64. A linear model can't capture the nonlinear effect of location, and the predicted-vs-actual plot bends away from the diagonal at both ends.
The 500,001 cap shows up as the vertical column in the predicted-vs-actual plot and the diagonal line in the residual plot. On test rows below the cap, OLS RMSE is 60,850 against 68,845 on all rows, so the cap inflates the error.
Some predictions are negative, which is impossible for a house value.
Residuals are right-skewed and their spread grows with the predicted value, which breaks the constant-variance assumption behind OLS inference.
One split (seed 42) produces the test numbers, and the ISLAND category is too small for any conclusion about it.
Files
task-2-california-housing/
├── task2_california_housing.ipynb    # full pipeline, run top to bottom
├── Real-estate-housing.csv           # dataset
└── outputs/
    ├── 01_correlation_heatmap.png
    ├── 02_alpha_curves.png
    ├── 03_predicted_vs_actual.png
    ├── 04_residual_distributions.png
    ├── coefficients.csv              # standardized coefficients per model
    └── results.csv                   # model comparison table
How to run
pandas
numpy
scikit-learn>=1.1
seaborn
matplotlib
jupyterlab
ipython

Content
Real-estate-housing.csv

CSV
jupyterlab
ipython



Needs scikit-learn 1.1 or newer for `DecisionBoundaryDisplay`.

