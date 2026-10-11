# mall-customer-spending-prediction
A Linear Regression project to analyze the relationship between annual income and spending score using Python and Scikit-learn.
# Mall Customer Spending Score Prediction using Linear Regression

#  Project Overview

This project uses **Linear Regression** to analyze the relationship between customers' annual income and their spending scores using the Mall Customers dataset.

The project demonstrates a basic Machine Learning workflow, including data exploration, model training, performance evaluation, and visualization.

#  Objectives

* Explore and understand the Mall Customers dataset.
* Analyze the relationship between Annual Income and Spending Score.
* Build a Linear Regression model using Scikit-learn.
* Evaluate model performance using multiple regression metrics.
* Visualize data distributions and regression results.
* Predict a spending score for a given annual income.

#  Dataset

The dataset used in this project was obtained from **Kaggle**.

It contains information about mall customers, including:

* Customer ID
* Gender
* Age
* Education
* Marital Status
* Annual Income (k$)
* Spending Score (1–100)

**Input Feature:** Annual Income (k$)
**Target Variable:** Spending Score (1–100)

The model uses Annual Income as its only input feature to predict the Spending Score.

**Dataset Source:** [Kaggle](https://www.kaggle.com/)



##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## ⚙️ Machine Learning Workflow

1. Load the dataset using Pandas.
2. Explore the data using `head()`, `info()`, and `describe()`.
3. Check missing values and duplicate rows.
4. Select Annual Income as the input feature and Spending Score as the target.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Train a Linear Regression model.
7. Evaluate predictions on the test dataset.
8. Visualize the spending score distribution and regression line.
9. Generate a prediction for a new annual income value.

#  Model Evaluation

The model was evaluated using the following metrics on the test dataset:

| Metric                         |  Result |
| ------------------------------ | ------: |
| Mean Absolute Error (MAE)      |   18.90 |
| Mean Squared Error (MSE)       |  573.95 |
| Root Mean Squared Error (RMSE) |   23.96 |
| R² Score                       | -0.1636 |

**Note:** These are the results from the current train-test split. The negative R² score indicates that this model performs worse than a baseline that predicts the mean spending score on this test set. Annual Income alone is not sufficient for reliable Spending Score prediction with this Linear Regression model.

#  Visualizations

The project includes:

* Spending Score Distribution Histogram
* Annual Income vs. Spending Score Scatter Plot
* Linear Regression Line
* Actual vs. Predicted Spending Score Plot

#  Example Prediction

The model can predict a Spending Score for a given Annual Income.

For example, an annual income of **$10,000** corresponds to an input value of `10` in the dataset's `Annual Income (k$)` column.

The model's current predicted Spending Score for this input is approximately **48.76**.

This is a model prediction, not a guarantee of a customer's actual spending behaviour.

#  Future Improvements

* Experiment with additional customer features.
* Compare Linear Regression with alternative regression algorithms.
* Analyze prediction errors and model assumptions.
* Compare model performance using consistent train-test splits.
* Explore K-Means Clustering for customer segmentation.

#  Key Learnings

* Exploratory Data Analysis (EDA)
* Linear Regression
* Train-Test Split
* Regression Evaluation Metrics
* Data Visualization
* Understanding Model Limitations

# 👨‍💻 Author

**Deepanshu**

GitHub: [deepanshurajpoot777-cpu](https://github.com/deepanshurajpoot777-cpu)
