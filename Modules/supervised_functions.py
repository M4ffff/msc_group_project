import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import make_classification,make_blobs
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder



def plot_decision_boundary(X, y, model, alpha=0.8, cmap='viridis'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    st.pyplot(fig)

def plot_decision_boundary_with_hyperplane(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='k', linestyles='--', levels=[-1, 0, 1], linewidths=1)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50, label='Data Points')

    w = model.coef_[0]
    b = model.intercept_[0]
    x_values = np.linspace(x_min, x_max, 100)
    y_values = -(w[0] * x_values + b) / w[1]

    ax.plot(x_values, y_values, color='black', linestyle='-', linewidth=2, label='Hyperplane')

    margin = 1 / np.sqrt(np.sum(w ** 2))
    y_values_upper = y_values + margin * (w[1] / np.linalg.norm(w))
    y_values_lower = y_values - margin * (w[1] / np.linalg.norm(w))

    ax.plot(x_values, y_values_upper, color='gray', linestyle='--', linewidth=1, label='Margin')
    ax.plot(x_values, y_values_lower, color='gray', linestyle='--', linewidth=1)

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            facecolors='none', edgecolors='black', s=120, linewidths=1.5, label='Support Vectors')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary with Hyperplane and Support Vectors')
    ax.legend()

    st.pyplot(fig)


# def about subpage
def subpage1(method):
    st.write(method)
    # Formula
    st.subheader("Formula")
    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")

    # data
    st.subheader("Data Generation")
    st.write("Let's generate some synthetic data to demonstrate linear regression.")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)

    np.random.seed(int(seed))
    X = 2.5 * np.random.rand(100, 1)
    y = 2 + 3 * X + np.random.randn(100, 1)

    # draw
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Points')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)

    # train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)


    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")


    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Points')
    ax.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    st.pyplot(fig)


    st.subheader("Interactive Prediction")
    x_value = st.slider("Choose a value for X", 0.0, 1.0, 0.5)
    y_pred = model.predict([[x_value]])
    st.write(f"Predicted y for X = {x_value}: {y_pred[0][0]:.2f}")


def subpage2(method):
    st.write(method)

    st.subheader("Formula")
    st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + ... + w_nx_n + b)}}")


    st.subheader("Data Generation")
    st.write("Let's generate some synthetic data to demonstrate logistic regression.")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed=int(seed)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1,
                            random_state=intseed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data Scatter Plot')
    st.pyplot(fig)


    model = LogisticRegression()
    model.fit(X_train, y_train)


    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    st.subheader("Decision Boundary")



    plot_decision_boundary(X_train, y_train, model)

    st.subheader("Interactive Prediction")
    feature1 = st.slider("Choose a value for Feature 1", X[:, 0].min(), X[:, 0].max(), (X[:, 0].mean()))
    feature2 = st.slider("Choose a value for Feature 2", X[:, 1].min(), X[:, 1].max(), (X[:, 1].mean()))
    input_features = np.array([[feature1, feature2]])
    prediction = model.predict(input_features)
    st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")


def subpage3(method):
    st.write(method)

    st.subheader("Formula")
    st.latex(r"f(x) = \text{sin}(w \cdot x + b)")

    st.subheader("Data Generation")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed=int(seed)
    X, y = make_blobs(n_samples=100, centers=2, random_state= intseed, cluster_std=1.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= intseed)

    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)


    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    st.subheader("Decision Boundary")


    # svc
    plot_decision_boundary(X_train, y_train, model)

    st.subheader("Interactive Prediction")
    feature1 = st.slider("Choose a value for Feature 1", float(X[:, 0].min()), float(X[:, 0].max()),
                        float(X[:, 0].mean()))
    feature2 = st.slider("Choose a value for Feature 2", float(X[:, 1].min()), float(X[:, 1].max()),
                        float(X[:, 1].mean()))
    input_features = np.array([[feature1, feature2]])
    prediction = model.predict(input_features)
    st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")

    st.subheader("Decision Boundary with Hyperplane and Support Vectors")



    plot_decision_boundary_with_hyperplane(X_train, y_train, model)


def subpage4(method):
    st.write(method)
    st.subheader("Data Generation")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed=int(seed)
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=2,
                            random_state=intseed, shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.2, random_state=  intseed)

    rf = RandomForestClassifier(n_estimators=100, random_state=intseed)
    rf.fit(X_train, y_train)

    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feature_importances = rf.feature_importances_
    feature_names = ['Feature 1', 'Feature 2']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_names)
    plt.title('Feature Importance')
    st.pyplot(plt)

    st.subheader("Model Evaluation")
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    st.subheader("Decision Boundary")



    plot_decision_boundary(X_train, y_train, rf, alpha=0.4, cmap='coolwarm')


        
        
        
def supervised_quiz():
    # Question 1: Linear Regression
    st.subheader("Question 1: Linear Regression")
    st.write("Which of the following statements is true about Linear Regression?")
    options_1 = ["A. Linear Regression is used for classification tasks.",
                    "B. Linear Regression models the relationship between a dependent variable and one or more independent variables using a linear function.",
                    "C. Linear Regression assumes that the data is not linearly separable.",
                    "D. Linear Regression cannot be used for predicting continuous values."]
    selected_option_1 = st.radio("Select your answer:", options_1, key="q1", index=None)


    if selected_option_1 is None:
        st.write("")
    elif selected_option_1 == options_1[1]:
        st.success("Correct! Linear Regression models the relationship using a linear function.")
    else:
        st.error("Incorrect. The correct answer is B.")
    
    
    
    # Question 2: Logistic Regression
    st.subheader("Question 2: Logistic Regression")
    st.write("Which of the following statements is true about Logistic Regression?")
    options_2 = ["A. Logistic Regression is used for predicting continuous numerical values.",
                    "B. Logistic Regression uses a linear function to model the relationship between variables.",
                    "C. Logistic Regression is a type of regression algorithm that outputs probabilities for classification tasks.",
                    "D. Logistic Regression assumes that the data is linearly separable."]
    selected_option_2 = st.radio("Select your answer:", options_2, key="q2",index=None)

    if selected_option_2 is None:
        st.write("")
    elif selected_option_2 == options_2[2]:
        st.success("Correct! Logistic Regression outputs probabilities for classification tasks.")
    else:
        st.error("Incorrect. The correct answer is C.")
        
        
    
    # Question 3: Support Vector Machine (SVM)
    st.subheader("Question 3: Support Vector Machine (SVM)")
    st.write("Which of the following statements is true about Support Vector Machine (SVM)?")
    options_3 = ["A. SVM is primarily used for unsupervised learning tasks.",
                    "B. SVM aims to find the hyperplane that maximizes the margin between two classes.",
                    "C. SVM cannot handle non-linear data without using kernel functions.",
                    "D. SVM is not suitable for classification tasks with high-dimensional data."]
    selected_option_3 = st.radio("Select your answer:", options_3, key="q3",index=None)

    if selected_option_3 is None:
        st.write("")
    elif selected_option_3 == options_3[1]:
        st.success("Correct! SVM aims to find the hyperplane that maximizes the margin between two classes.")
    else:
        st.error("Incorrect. The correct answer is B.")
    
    
    
    st.subheader("Question 4: Random Forest")
    st.write("Which of the following statements is true about Random Forest?")
    options_4 = ["A. Random Forest is a type of neural network.",
                    "B. Random Forest builds multiple decision trees and merges them to get a more accurate prediction.",
                    "C. Random Forest is not suitable for classification tasks.",
                    "D. Random Forest requires a large amount of data to be effective."]
    selected_option_4 = st.radio("Select your answer:", options_4, key="q4",index=None)

    if selected_option_4 is None:
        st.write("")
    elif selected_option_4 == options_4[1]:
        st.success("Correct! Random Forest builds multiple decision trees for more accurate predictions.")
    else:
        st.error("Incorrect. The correct answer is B.")
        