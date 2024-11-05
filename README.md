# HR Management - Employee Retention Prediction

## Project Overview

Employee retention is crucial for maintaining a productive and experienced workforce. In this project, we aim to predict the likelihood of an employee leaving or staying at a company based on various factors such as **distance from the company**, **average salary**, **days off from work**, and more. By identifying the drivers behind employee turnover, HR departments can take proactive steps to improve retention and reduce associated costs.

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Dataset Overview](#dataset-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Models Implemented](#models-implemented)
   - [Logistic Regression](#logistic-regression)
   - [Random Forest Classifier](#random-forest-classifier)
   - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Installation & Usage](#installation--usage)

## Project Objectives

- **Predict** whether an employee will leave or stay in the company.
- **Analyze** the key factors influencing employee attrition.
- **Develop a model** to assist HR departments in proactive decision-making for talent retention.

## Dataset Overview

The dataset consists of employee information from various departments, roles, and personal attributes. Key features include:
- **DistanceFromHome**: Employee's distance from the company.
- **MonthlyIncome**: Employee’s monthly income.
- **YearsAtCompany**: Tenure at the company.
- **OverTime**: Overtime status.
- **JobSatisfaction, WorkLifeBalance, JobRole**, and more.

The target variable, **Attrition**, indicates if an employee has left the company (1) or stayed (0).

## Exploratory Data Analysis (EDA)

EDA was conducted to better understand the distribution and relationships of features within the dataset:
- **Histograms** for numerical features to understand their distributions.
- **Count plots** for categorical features to observe trends in attrition.
- **Correlation heatmap** to identify relationships between variables (e.g., `JobLevel` and `MonthlyIncome` showed strong correlation).
- **KDE plots** to analyze the distribution of continuous variables like `DistanceFromHome` and `TotalWorkingYears`.

## Data Preprocessing

Before modeling, data preprocessing steps were applied:
- **Encoding categorical variables**: Used one-hot encoding for columns like `BusinessTravel`, `Department`, `EducationField`, `Gender`, etc.
- **Scaling numerical features**: Used MinMaxScaler to normalize features and ensure equal contribution across models.
- **Handling missing values**: Verified no missing data through heatmap analysis.

## Models Implemented

### Logistic Regression

A logistic regression model was trained to provide an interpretable baseline for predicting employee attrition. Logistic regression was selected due to its simplicity and interpretability, making it suitable for explaining the impact of each feature on employee attrition.

- **Accuracy**: 85.6%
- **Confusion Matrix** and **Classification Report** used to evaluate performance.

### Random Forest Classifier

To capture non-linear relationships, a Random Forest Classifier was trained. This model offers feature importance insights, helping HR understand the most influential factors affecting attrition.

- **Accuracy**: 86%
- **Feature Importance Analysis** was conducted to identify key predictors.

### Artificial Neural Network (ANN)

A deep learning model (ANN) was also built to improve prediction accuracy by learning complex patterns in the data. The architecture consisted of:
- **3 hidden layers** with ReLU activation, each with 500 neurons.
- **Output layer** with sigmoid activation for binary classification.
  
- **Loss function**: Binary Cross-Entropy.
- **Optimizer**: Adam.

The ANN achieved high accuracy over 100 epochs, showcasing the potential of deep learning for HR analytics.

## Evaluation Metrics

Each model was evaluated using:
- **Accuracy**: The overall correctness of predictions.
- **Precision, Recall, F1-Score**: Especially useful for the attrition class, given the imbalanced nature of the dataset.
- **Confusion Matrix**: To understand the model's performance in distinguishing between employees who left and those who stayed.

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 85.6%    | 0.84      | 0.86   | 0.84     |
| Random Forest Classifier | 86%      | 0.88      | 0.86   | 0.82     |
| ANN                      | 85%      | 0.82      | 0.85   | 0.83     |

## Results

The Random Forest model provided the best performance in terms of accuracy and interpretability, making it suitable for understanding employee retention trends. However, the ANN model showed potential for higher accuracy with more data and tuning.

- **Key Features Impacting Attrition**: 
  - **DistanceFromHome**: Higher distances showed correlation with attrition.
  - **MonthlyIncome and JobSatisfaction**: Higher income and job satisfaction correlated with lower attrition.
  - **OverTime**: Employees working overtime were more likely to leave.

## Conclusion

The models successfully demonstrated the potential of machine learning in predicting employee attrition. Insights from this project could aid HR departments in understanding key retention factors and implementing data-driven strategies to improve employee satisfaction and reduce turnover.

Future work could involve hyperparameter tuning, balancing class weights, and exploring additional models to further enhance performance.

## Installation & Usage

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `tensorflow`

### Installation and Usage
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/HR_Management_Retention_Prediction.git
cd HR_Management_Retention_Prediction
pip install -r requirements.txt
jupyter notebook HR_Management.ipynb



