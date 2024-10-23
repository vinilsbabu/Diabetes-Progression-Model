# Diabetes Progression Model

## Objective
This project aims to model the progression of diabetes using machine learning techniques, specifically an Artificial Neural Network (ANN). The model utilizes the Diabetes dataset from the `sklearn` library to understand how various factors influence diabetes progression, providing insights that can aid healthcare professionals in designing better treatment plans and preventive measures.

## Dataset
The dataset used in this project is the Diabetes dataset from the `sklearn` library. It contains features that describe various health measurements of individuals along with the target variable indicating the progression of diabetes.

### Key Findings
- The dataset does not have any missing data.
- Glucose levels had the highest effect on the outcome.
- As expected, pregnancies were correlated to age.

## Key Components

1. **Loading and Preprocessing**
    - Load the Diabetes dataset.
    - Normalize features for better performance of the ANN model.

2. **Exploratory Data Analysis (EDA)**
    - Analyze the distribution of features and the target variable.
    - Visualize relationships between features and the target variable.

3. **Building the ANN Model**
    - Design a simple ANN architecture with at least one hidden layer.
    - Use appropriate activation functions.

4. **Training the ANN Model**
    - Split the dataset into training and testing sets.
    - Train the model using the training data with a suitable loss function and optimizer.

5. **Evaluating the Model**
    - Evaluate the model on the testing data.
    - Performance metrics:
        - **Mean Squared Error**: 2813.13
        - **R² Score**: 0.47

6. **Improving the Model**
    - Experiment with different architectures and hyperparameters.
    - After experimentation:
        - **Improved Mean Squared Error**: 3729.71
        - **Improved R² Score**: 0.30

## Requirements
To run this project, you will need the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
