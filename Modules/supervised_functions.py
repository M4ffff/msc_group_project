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


#
def plot_decision_boundary(X, y, model, alpha=0.8, cmap='viridis'):
    """
       Plot the decision boundary of a classification model on a 2D feature space.

       Parameters:
       - X: numpy array of shape (n_samples), the feature data.
       - y: numpy array of shape (n_samples), the target labels.
       - model: a trained classification model with a predict method.
       - alpha: float, the transparency level of the contour plot (default=0.8).
       - cmap: str, the colormap for the decision boundary (default='viridis').

       This function creates a contour plot of the decision boundary and overlays
       the training data points colored by their class labels.
       """
    # Determine the range of the feature space for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Range for the first feature
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Range for the second feature

    # Create a meshgrid of points within the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),  # Grid for the first feature
                         np.arange(y_min, y_max, 0.1))  # Grid for the second feature

    # Predict the class labels for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Flatten the grid and predict
    Z = Z.reshape(xx.shape)  # Reshape the predictions to match the grid shape

    # Create a figure with a specified size
    fig = plt.figure(figsize=(8, 6))

    # Plot the decision boundary as a contour plot
    plt.contourf(xx, yy, Z, alpha=alpha)  # Fill the contour with colors

    # Overlay the training data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')  # Scatter plot of data points

    # Add labels and title to the plot
    plt.xlabel('Feature 1')  # Label for the first feature
    plt.ylabel('Feature 2')  # Label for the second feature
    plt.title('Decision Boundary')  # Title of the plot

    # Display the plot using Streamlit
    st.pyplot(fig)

def plot_decision_boundary_with_hyperplane(X, y, model):
    """
    Plot the decision boundary, hyperplane, and support vectors of an SVM model on a 2D feature space.

    Parameters:
    - X: numpy array of shape (n_samples, ), the feature data.
    - y: numpy array of shape (n_samples,), the target labels.
    - model: a trained SVM model with attributes coef_ and intercept_.

    This function visualizes the decision boundary, the hyperplane, and the support vectors of the SVM model.
    It also includes the margin boundaries and labels the support vectors.
    """
    # Determine the range of the feature space for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Range for the first feature
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Range for the second feature

    # Create a meshgrid of points within the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),  # Grid for the first feature
                         np.arange(y_min, y_max, 0.1))  # Grid for the second feature

    # Predict the class labels for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Flatten the grid and predict
    Z = Z.reshape(xx.shape)  # Reshape the predictions to match the grid shape

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the decision boundary as a contour plot with filled colors
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')  # Filled contour plot

    # Plot the decision boundary as a contour plot with lines
    ax.contour(xx, yy, Z, colors='k', linestyles='--', levels=[-1, 0, 1], linewidths=1)  # Decision boundary lines

    # Overlay the training data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50, label='Data Points')  # Scatter plot of data points

    # Extract the coefficients and intercept from the SVM model
    w = model.coef_[0]  # Coefficients of the hyperplane
    b = model.intercept_[0]  # Intercept of the hyperplane

    # Generate x values for plotting the hyperplane
    x_values = np.linspace(x_min, x_max, 100)  # Linearly spaced x values

    # Calculate corresponding y values for the hyperplane
    y_values = -(w[0] * x_values + b) / w[1]  # Hyperplane equation

    # Plot the hyperplane
    ax.plot(x_values, y_values, color='black', linestyle='-', linewidth=2, label='Hyperplane')  # Hyperplane line

    # Calculate the margin boundaries
    margin = 1 / np.sqrt(np.sum(w ** 2))  # Margin width
    y_values_upper = y_values + margin * (w[1] / np.linalg.norm(w))  # Upper margin boundary
    y_values_lower = y_values - margin * (w[1] / np.linalg.norm(w))  # Lower margin boundary

    # Plot the margin boundaries
    ax.plot(x_values, y_values_upper, color='gray', linestyle='--', linewidth=1, label='Margin')  # Upper margin line
    ax.plot(x_values, y_values_lower, color='gray', linestyle='--', linewidth=1)  # Lower margin line

    # Plot the support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               facecolors='none', edgecolors='black', s=120, linewidths=1.5, label='Support Vectors')  # Support vectors

    # Add labels and title to the plot
    ax.set_xlabel('Feature 1')  # Label for the first feature
    ax.set_ylabel('Feature 2')  # Label for the second feature
    ax.set_title('Decision Boundary with Hyperplane and Support Vectors')  # Title of the plot
    ax.legend()  # Add legend to the plot

    # Display the plot using Streamlit
    st.pyplot(fig)


# Define the subpage function
def subpage1(method):
    """
    This function creates a Streamlit subpage to demonstrate linear regression.
    It includes sections for formula explanation, data generation, visualization,
    model training, evaluation, and interactive prediction.

    Parameters:
    - method: str, the method name to be displayed on the subpage.
    """

    # Display the method name
    st.write(method)

    # Formula section
    st.subheader("Formula")
    with st.expander("**TOOL PURPOSE:**"):
        st.markdown(
            """
        This interactive Neural Network Tool enables users of all skill levels to configure, visualise, and explore
        there own neural network models through an intuitive interface. Navigate the following tabs to learn about various 
        components and parameters of neural networks, and examine how they influence the model's behaviour/outcomes
        and classification performance. This tool offers a hands-on, visual learning experience, introducing users to neural
        networks without the need for time-consuming programming. If you want to learn about the code, you can do that too, 
        under the **Python Implementation** tab.
        """)
    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")
    # Explanation of the linear regression formula

    # Data Generation section
    st.subheader("Data Generation")
    with st.expander("**TOOL PURPOSE:**"):
        st.markdown(
            """
        This interactive Neural Network Tool enables users of all skill levels to configure, visualise, and explore
        there own neural network models through an intuitive interface. Navigate the following tabs to learn about various 
        components and parameters of neural networks, and examine how they influence the model's behaviour/outcomes
        and classification performance. This tool offers a hands-on, visual learning experience, introducing users to neural
        networks without the need for time-consuming programming. If you want to learn about the code, you can do that too, 
        under the **Python Implementation** tab.
        """)
    st.write("Let's generate some synthetic data to demonstrate linear regression.")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    # Slider to choose a random seed for reproducibility

    # Generate synthetic data
    np.random.seed(int(seed))
    X = 2.5 * np.random.rand(100, 1)  # Random feature values
    y = 2 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

    # Data Visualization section
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Points')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)
    # Visualize the generated data points

    # Model Training section
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the data into training and testing sets
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Train a linear regression model

    # Model Evaluation section
    st.subheader("Model Evaluation")
    with st.expander("**TOOL PURPOSE:**"):
        st.markdown(
            """
        This interactive Neural Network Tool enables users of all skill levels to configure, visualise, and explore
        there own neural network models through an intuitive interface. Navigate the following tabs to learn about various 
        components and parameters of neural networks, and examine how they influence the model's behaviour/outcomes
        and classification performance. This tool offers a hands-on, visual learning experience, introducing users to neural
        networks without the need for time-consuming programming. If you want to learn about the code, you can do that too, 
        under the **Python Implementation** tab.
        """)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    # Evaluate the model using Mean Squared Error

    # Visualize the regression line
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Data Points')
    ax.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    st.pyplot(fig)
    # Plot the data points and the regression line

    # Interactive Prediction section
    st.subheader("Interactive Prediction")
    with st.expander("**TOOL PURPOSE:**"):
        st.markdown(
            """
        This interactive Neural Network Tool enables users of all skill levels to configure, visualise, and explore
        there own neural network models through an intuitive interface. Navigate the following tabs to learn about various 
        components and parameters of neural networks, and examine how they influence the model's behaviour/outcomes
        and classification performance. This tool offers a hands-on, visual learning experience, introducing users to neural
        networks without the need for time-consuming programming. If you want to learn about the code, you can do that too, 
        under the **Python Implementation** tab.
        """)
    x_value = st.slider("Choose a value for X", 0.0, 1.0, 0.5)
    # Slider to choose a value for X
    y_pred = model.predict([[x_value]])
    st.write(f"Predicted y for X = {x_value}: {y_pred[0][0]:.2f}")
    # Predict and display the corresponding y value

# Define the subpage function
def subpage2(method):
    """
    This function creates a Streamlit subpage to demonstrate logistic regression.
    It includes sections for formula explanation, data generation, visualization,
    model training, evaluation, decision boundary plotting, and interactive prediction.

    Parameters:
    - method: str, the method name to be displayed on the subpage.
    """

    # Display the method name
    st.write(method)

    # Formula section
    st.subheader("Formula")
    st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + ... + w_nx_n + b)}}")
    # Explanation of the logistic regression formula

    # Data Generation section
    st.subheader("Data Generation")
    st.write("Let's generate some synthetic data to demonstrate logistic regression.")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed = int(seed)

    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1,
                               random_state=intseed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the data into training and testing sets

    # Data Visualization section
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data Scatter Plot')
    st.pyplot(fig)
    # Visualize the training data points

    # Model Training section
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Train a logistic regression model

    # Model Evaluation section
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    # Evaluate the model using accuracy and confusion matrix

    # Decision Boundary section
    st.subheader("Decision Boundary")

    plot_decision_boundary(X_train, y_train, model)
    # Visualize the decision boundary of the logistic regression model

    # Interactive Prediction section
    st.subheader("Interactive Prediction")
    feature1 = st.slider("Choose a value for Feature 1", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    feature2 = st.slider("Choose a value for Feature 2", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    input_features = np.array([[feature1, feature2]])
    prediction = model.predict(input_features)
    st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")
    # Allow users to input feature values and get predictions


# Define the subpage function
def subpage3(method):
    """
    This function creates a Streamlit subpage to demonstrate Support Vector Machines (SVM) with a linear kernel.
    It includes sections for formula explanation, data generation, visualization, model training, evaluation,
    decision boundary plotting, interactive prediction, and a detailed visualization of the decision boundary
    with the hyperplane and support vectors.

    Parameters:
    - method: str, the method name to be displayed on the subpage.
    """

    # Display the method name
    st.write(method)

    # Formula section
    st.subheader("Formula")
    st.latex(r"f(x) = \text{sin}(w \cdot x + b)")
    # Explanation of the SVM formula (Note: This is a placeholder formula; SVMs use a different formula.)

    # Data Generation section
    st.subheader("Data Generation")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed = int(seed)

    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=2, random_state=intseed, cluster_std=1.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=intseed)
    # Split the data into training and testing sets

    # Data Visualization section
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)
    # Visualize the data points

    # Model Training section
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    # Train an SVM model with a linear kernel

    # Model Evaluation section
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    # Evaluate the model using accuracy

    # Confusion Matrix section
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    # Visualize the confusion matrix using a heatmap

    # Decision Boundary section
    st.subheader("Decision Boundary")

    # Plot decision boundary
    def plot_decision_boundary(X, y, model):
        """
        Plot the decision boundary of an SVM model.
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        st.pyplot(fig)

    plot_decision_boundary(X_train, y_train, model)
    # Visualize the decision boundary of the SVM model

    # Interactive Prediction section
    st.subheader("Interactive Prediction")
    feature1 = st.slider("Choose a value for Feature 1", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    feature2 = st.slider("Choose a value for Feature 2", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    input_features = np.array([[feature1, feature2]])
    prediction = model.predict(input_features)
    st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")
    # Allow users to input feature values and get predictions

    # Decision Boundary with Hyperplane and Support Vectors section
    st.subheader("Decision Boundary with Hyperplane and Support Vectors")

    # Plot decision boundary with hyperplane and support vectors
    plot_decision_boundary_with_hyperplane(X_train, y_train, model)
    # Visualize the decision boundary with the hyperplane and support vectors


def subpage4(method):
    """
    This function creates a Streamlit subpage to demonstrate the Random Forest Classifier.
    It includes sections for data generation, visualization, feature importance analysis,
    model evaluation, and decision boundary plotting.

    Parameters:
    - method: str, the method name to be displayed on the subpage.
    """

    # Display the method name
    st.write(method)

    # Data Generation section
    st.subheader("Data Generation")
    seed = st.slider("Choose a random seed", 0.0, 100.0, 50.0)
    intseed = int(seed)

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=2,
                               random_state=intseed, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X[:, :2], y, test_size=0.2, random_state=intseed)
    # Split the data into training and testing sets, using only the first two features for visualization

    # Model Training section
    rf = RandomForestClassifier(n_estimators=100, random_state=intseed)
    rf.fit(X_train, y_train)
    # Train a Random Forest Classifier

    # Data Visualization section
    st.subheader("Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of X vs y')
    st.pyplot(fig)
    # Visualize the data points

    # Feature Importance section
    st.subheader("Feature Importance")
    feature_importances = rf.feature_importances_
    feature_names = ['Feature 1', 'Feature 2']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_names)
    plt.title('Feature Importance')
    st.pyplot(plt)
    # Visualize the feature importances using a bar plot

    # Model Evaluation section
    st.subheader("Model Evaluation")
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    # Evaluate the model using a confusion matrix

    # Decision Boundary section
    st.subheader("Decision Boundary")

    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, rf, alpha=0.4, cmap='coolwarm')
    # Visualize the decision boundary of the Random Forest model


        
        
        
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
        