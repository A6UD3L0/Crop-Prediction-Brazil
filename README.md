
# Project Introduction

Welcome to the data science project developed during the 5th semester of university for a competition hosted by Bain & Company. This project aimed to minimize the quadratic square error in predictive models. The following two scripts showcase my early exploration into data science, covering a comprehensive data science pipeline and the implementation of a SuperLearner ensemble.

## Data Science Pipeline (DataPipeline.py)

### Overview
The **DataPipeline.py** script implements a robust data science pipeline, covering data loading, classification, one-hot encoding, Gower distance matrix calculation, outlier detection, and missing data imputation. The primary goal was to preprocess and prepare data for predictive modeling, adhering to the competition's requirement of minimizing the quadratic square error.

### Features
1. **Data Loading and Classification:**
   - Loads data from a CSV file and classifies variables, including converting specific columns to categorical types.

2. **One-Hot Encoding:**
   - Assigns categorical variables using One-Hot Encoding and adds dummy columns for categorical features.

3. **Gower Distance Matrix Calculation:**
   - Utilizes the Gower distance matrix for both categorical and numerical features.

4. **Outlier Detection:**
   - Detects outliers using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and IQR (Interquartile Range) methods.

5. **Missing Data Imputation:**
   - Implements MissForest for imputing missing data.

6. **Main Execution:**
   - Executes the entire pipeline on a given dataset.

### How to Use
1. **DataPipeline.py:**
   - Ensure all required libraries are installed (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `gower`, `missingpy`).
   - Set the correct file path in the `main` function.
   - Run the script to execute the entire data science pipeline.

## SuperLearner Ensemble (SuperLearner.py)

### Overview
The **SuperLearner.py** script demonstrates the use of a SuperLearner ensemble with various base classifiers for predicting 'destinated_area' in a dataset. The base models include Logistic Regression, Decision Tree, SVM, Naive Bayes, K-Nearest Neighbors, AdaBoost, Bagging, Random Forest, and Extra Trees classifiers.

### Features
1. **Base Models:**
   - Utilizes various base classifiers to form the SuperLearner ensemble.

2. **Ensemble Configuration:**
   - Implements a SuperLearner ensemble with a meta-model (Logistic Regression).
   - Evaluates the ensemble on a validation set and prints the accuracy score.

### How to Use
1. **SuperLearner.py:**
   - Ensure all required libraries are installed (`scikit-learn`, `mlens`).
   - Ensure the 'Categorical_values' module is available.
   - Load data using the provided function from 'Categorical_values'.
   - Run the script to create a SuperLearner ensemble and evaluate its performance.

## Notes
- This project represents my early exploration into data science during the 5th semester of university.
- For additional details, refer to the script comments and the competition context provided.
"""
