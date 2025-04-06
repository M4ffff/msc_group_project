import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
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
import streamlit.components.v1 as components

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
    with st.expander("**Components of the Formula:**"):
        st.markdown(
            """
        y: This represents the dependent variable, also known as the target variable or the outcome that we want to predict or explain.\n
        β0: This is the intercept term. It is the value of y when all the independent variables (x 1,x2,…,xn) are zero. It essentially represents the baseline level of the dependent variable.\n
        β1,β2,…,βn:These are the coefficients for the independent variables x1,x2,…,xnEach coefficient indicates the change in the dependent variable y for a one-unit change in the corresponding independent variable, holding all other variables constant.\n
        x1,x2,…,x n: These are the independent variables, also known as predictors or features. They are the inputs used to predict the dependent variable.\n
        ϵ: This is the error term, which represents the difference between the observed value of y and the value predicted by the model. It accounts for the variability in y that cannot be explained by the independent variables.\n
        """)
    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")
    with st.expander("**Principle of Linear Regression:**"):
        st.markdown(
            """
        Linear regression aims to find the best-fitting straight line (in the case of simple linear regression with one independent variable) or hyperplane (in the case of multiple linear regression with multiple independent variables) through the data points. The goal is to minimize the sum of the squared differences between the observed values of y and the values predicted by the model. This process is known as ordinary least squares (OLS) estimation.\n
        Here’s how it works:\n
        Model Specification: We start by specifying the linear model as shown in the formula. We assume that the relationship between the dependent variable and the independent variables is linear.\n
        Estimation of Coefficients: Using the data, we estimate the values of the coefficients (β 0,β1,…,β n) that minimize the sum of the squared errors (ϵ). This is done through a mathematical optimization process.\n
        Prediction: Once the coefficients are estimated, we can use the model to predict the value of y for new values of the independent variables.\n
        Evaluation: We evaluate the model’s performance by examining metrics such as the coefficient of determination (R2), which indicates the proportion of the variance in the dependent variable that is predictable from the independent variables, and the significance of the coefficients.\n
         In summary, linear regression is a powerful tool for understanding and predicting the relationship between variables by fitting a linear model to the data.
        """)
    # Explanation of the linear regression formula

    # Data Generation section
    st.subheader("Data Generation")
    with st.expander("**How we get random data:**"):
        st.code("np.random.seed(int(seed))")
        st.markdown(
            """
        Purpose:\n
        Sets the seed for the random number generator.\n
        Explanation:\n
        By setting a seed, the random numbers generated will be reproducible. This means that every time you run the code with the same seed, you will get the same sequence of random numbers. This is useful for debugging and ensuring consistent results.\n
        """)
        st.code("X/Y= 2.5 * np.random.rand(100, 1)  # Random feature values")
        st.markdown(
            """
            Purpose: \n
            Generates random feature values for the independent variable X.\n
            Explanation:\n
            np.random.rand(100, 1) generates 100 random numbers between 0 and 1, forming a 100x1 array (matrix).\n
            Multiplying by 2.5 scales these random numbers to a range between 0 and 2.5. This creates the feature values for X.\n
            """)

    st.write("Let's generate some synthetic data to demonstrate linear regression.")
    seed = st.slider("Choose a random seed", 0, 100, 1)
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
    with st.expander("**What we have done:**"):
        st.markdown(
            """
       #### Splitting the Data:\n
       """)
        st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        st.markdown(
            """
            The dataset is divided into training and testing sets using the train_test_split function.\n
            X and y are the feature and target arrays, respectively.\n
            test_size=0.2 means that 20% of the data is used for testing, while the remaining 80% is used for training.\n
            random_state=42 ensures reproducibility by setting a fixed seed for the random splitting process.\n
            ##### Training the Model:\n
            """)
        st.code("""    
            model = LinearRegression()\n
            model.fit(X_train, y_train)\n
               """)
        st.markdown(
                     """
             A LinearRegression model is created.\n
             The model is trained using the training data (X_train and y_train) by calling the fit method. This step finds the best-fitting line (or hyperplane) that minimizes the error between the predicted and actual values in the training set.\n
              #### Summary\n
              This code splits the data into training and testing sets and trains a linear regression model on the training data. The model is then ready to make predictions on the test set, which is used to evaluate its performance.
""")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    with st.expander("**What is Mean Squared Error:**"):
        st.markdown(
            """
         Mean Squared Error is a widely used measure of the difference between the estimated values and the actual values in statistics and machine learning. Here is a detailed explanation:\n
         #### Definition\n
         Mean Squared Error is the average of the squares of the differences between the predicted values and the actual values. It quantifies the average squared magnitude of the errors.\n
         #### Formula \n
        If we have a set of n observations, where yi represents the actual value and  y^i represents the predicted value for the i-th observation, the Mean Squared Error (MSE) is calculated as: \n
             """)
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        st.markdown(
            """
        is the error for the i-th observation, and squaring this error ensures that all errors are positive and magnifies larger errors.\n
         #### Properties\n
        Non-negativity: MSE is always non-negative because it involves squaring the differences. A value of zero indicates a perfect fit where all predictions match the actual values exactly.\n
        Sensitivity to outliers: Since MSE involves squaring the errors, it is more sensitive to larger errors. This means that a few large errors can significantly increase the overall MSE value.\n
        Units: The units of MSE are the square of the units of the dependent variable. For example, if the dependent variable is measured in meters, the MSE will be in square meters.\n
         #### Applications\n
        Regression Analysis: MSE is commonly used as a loss function in regression models to evaluate the performance of the model. It helps in optimizing the model parameters to minimize the prediction errors.\n
        Model Selection: In model selection, MSE can be used to compare different models. A model with a lower MSE is generally considered to have better predictive accuracy.\n
        Forecasting: In time series forecasting, MSE is used to assess the accuracy of forecasts by comparing the predicted values with the actual values over time.\n
         #### Limitations\n
        Interpretability: The squared units of MSE can make it difficult to interpret directly. For example, it may not be immediately clear how an MSE of 100 compares to an MSE of 1000 in terms of practical significance.\n
        Overemphasis on large errors: Because MSE squares the errors, it may overemphasize the impact of outliers or large errors. In some cases, other metrics like Mean Absolute Error (MAE) might be more appropriate if the goal is to minimize the average error magnitude rather than the squared error.\n
        In summary, Mean Squared Error is a fundamental metric used to evaluate the accuracy of predictions in various fields such as statistics, machine learning, and econometrics. Its simplicity and mathematical properties make it a popular choice, but it is important to consider its limitations and choose the appropriate metric based on the specific problem and data characteristics.\n
              """)
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
    with st.expander("**PURPOSE:**"):
        st.markdown(
        """
        Just use the model to predict a X value and get the corresponding y value.
        """)
    x_value = st.slider("Choose a value for X", 0.0, 1.0, 0.5)
    # Slider to choose a value for X
    y_pred = model.predict([[x_value]])
    st.write(f"Predicted y for X = {x_value}: {y_pred[0][0]:.2f}")
    # Predict and display the corresponding y value

    # Part 1
    st.subheader("Do you finish it?")
    if st.button("Yes"):
        st.session_state.part1_visited = True


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
    with st.expander("**Components of the Formula:**"):
        st.markdown(
            """
        P(y=1∣x): This is the probability that the target variable y is 1 given the input features x. In other words, it is the predicted probability of the positive class.\n
        w1,w2,…,wn: These are the coefficients (or weights) associated with each input feature x1,x2,…,xn. These coefficients are learned during the training process.\n
        b: This is the bias term (or intercept). It is a constant term that allows the model to fit the data better by shifting the decision boundary.\n
        e: This is the base of the natural logarithm, approximately equal to 2.71828.\n
        """)
    st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + ... + w_nx_n + b)}}")
    with st.expander("**Principle of Logistic Regression:**"):
        st.markdown(
            """
        #### 1.Linear Combination: \n
        The input features x1,x2,…,xn are combined linearly with their respective coefficients w1,w2,…,wn,and the bias term b is added. This results in a linear score:z=w1x1+w2x2+…+wnxn+b\n
        #### 2.Logistic Function:\n 
        The linear score z is then passed through the logistic function (also known as the sigmoid function):\n
        The logistic function maps the linear score z to a value between 0 and 1, which can be interpreted as a probability.\n
        As z approaches positive infinity, P(y=1∣x) approaches 1.\n
        As z approaches negative infinity, P(y=1∣x) approaches 0.\n
         When z=0, P(y=1∣x)=0.5.\n
         #### Summary\n
         The formula represents the logistic regression model, which is used for binary classification problems. It calculates the probability that the target variable y is 1 given the input features x by applying the logistic function to a linear combination of the features and the bias term.
        """)
    #### Explanation of the logistic regression formula

    # Data Generation section
    st.subheader("Data Generation")
    with st.expander("**How we get random data:**"):
        st.code("np.random.seed(int(seed))")
        st.markdown(
        """
        Purpose:\n
        Sets the seed for the random number generator.\n
        Explanation:\n
        By setting a seed, the random numbers generated will be reproducible. This means that every time you run the code with the same seed, you will get the same sequence of random numbers. This is useful for debugging and ensuring consistent results.\n
        """)
        st.code("X, y = make_classification(n_samples=100, n_features=2,n_redundant=0, n_clusters_per_class=1, flip_y=0.1,random_state=intseed)")
        st.markdown("""
        Purpose: \n
        Generates random feature values for the independent variable X.\n
        Explanation:\n
        Synthetic classification data is generated with 100 samples and 2 features. The data is designed to be linearly separable with minimal redundancy and some label noise (flip_y=0.1).\n
         """)
    st.write("Let's generate some synthetic data to demonstrate logistic regression.")
    seed = st.slider("Choose a random seed", 0, 100, 1)
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
    with st.expander("**What we have done:**"):
       st.markdown(
       """
       #### Data Generation:\n
       """)
       st.code("X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1, random_state=intseed)")
       st.markdown(
       """
       Synthetic classification data is generated with 100 samples and 2 features. The data is designed to be linearly separable with minimal redundancy and some label noise (flip_y=0.1).Synthetic classification data is generated with 100 samples and 2 features. The data is designed to be linearly separable with minimal redundancy and some label noise (flip_y=0.1).\n
       #### Splitting the Data:\n
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n
       The dataset is divided into training and testing sets using the train_test_split function.\n
       X and y are the feature and target arrays, respectively.\n
       test_size=0.2 means that 20% of the data is used for testing, while the remaining 80% is used for training.\n
       random_state=42 ensures reproducibility by setting a fixed seed for the random splitting process.\n
       #### Training the Model:\n
           model = LogisticRegression()\n
           model.fit(X_train, y_train)\n
       A logistic regression model is trained on the training data..\n
        """)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    with st.expander("**What is Accuracy:**"):
        st.markdown(
        """
        #### Definition:\n 
        Accuracy is a measure of how often the model makes the correct prediction. It is the ratio of the number of correct predictions to the total number of predictions made.\n
        #### Formula: 
        If TP is the number of true positives, TN is the number of true negatives, FP is the number of false positives, and FN is the number of false negatives, then accuracy is calculated as:\n
        """)
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.markdown(
        """
        Explanation: In the case, Accuracy:0.XX  means that the model has an accuracy of XX%, indicating that the account of predictions made by the model are correct.
        """)
    st.write(f"Accuracy: {accuracy:.2f}")
    with st.expander("**What is Confusion Matrix:**"):
        st.latex(r"\begin{array}{c|cc}& \text{Predicted Positive} & \text{Predicted Negative} \\\hline\text{Actual Positive} & TP & FN \\\text{Actual Negative} & FP & TN \\\end{array}")
        st.markdown(
        """
        #### Definition:\n
        A confusion matrix is a table used to evaluate the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.\n
        #### Structure:\n
        True Positives (TP): The number of instances that were correctly predicted as positive.\n
        True Negatives (TN): The number of instances that were correctly predicted as negative.\n
        False Positives (FP): The number of instances that were incorrectly predicted as positive (Type I error).\n
        False Negatives (FN): The number of instances that were incorrectly predicted as negative (Type II error).\n
        """)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    # Evaluate the model using accuracy and confusion matrix

    # Decision Boundary section
    st.subheader("Decision Boundary")
    with st.expander("**What is Decision Boundary:**"):
        st.markdown(
        """
        #### Definition of Decision Boundary:\n
        A decision boundary is a boundary line or hyper-surface in a classification problem that separates different classes. It divides the feature space into multiple regions, each corresponding to a specific class. The model uses the decision boundary to determine the class of a sample based on its position in the feature space.\n
        In a geometric sense, the decision boundary is a hyper-surface. For linear classification problems, the decision boundary is a hyper-plane. For example, in a two-dimensional feature space, the decision boundary is a straight line; in a three-dimensional space, it is a plane.\n
        #### Decision Boundaries in Different Supervised Learning Algorithms:\n
        Although the essence of the decision boundary is to separate different classes, its shape and properties can vary across different supervised learning algorithms:\n
        #### Linear Models:\n
        Logistic Regression: The decision boundary is linear, defined by a hyper-plane that divides the feature space into two classes. For example, in a two-dimensional space, the decision boundary is a straight line.\n
        """)

    plot_decision_boundary(X_train, y_train, model)
    # Visualize the decision boundary of the logistic regression model

    # Interactive Prediction section
    st.subheader("Interactive Prediction")
    with st.expander("**PURPOSE:**"):
        st.markdown(
        """
        Just use the model to predict a X value and get the corresponding y value.
        """)
    feature1 = st.slider("Choose a value for Feature 1", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    feature2 = st.slider("Choose a value for Feature 2", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    input_features = np.array([[feature1, feature2]])
    prediction = model.predict(input_features)
    st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")
    # Allow users to input feature values and get predictions

    # Part 2
    st.subheader("Do you finish it?")
    if st.button("Yes"):
        st.session_state.part2_visited = True


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
    with st.expander("**Components of the Formula:**"):
        st.markdown(
        """
        f(x): This represents the function of x.\n
        w is the normal vector to the hyperplane (it determines the direction of the hyperplane).\n
        x is the input feature vector.\n
        b is the bias term (it determines the position of the hyperplane).\n
        sign is the sign function, which returns +1 for positive values and -1 for negative values.\n
        """)
    st.latex(r"f(x)=sign(w⋅x+b)")
    with st.expander("**Principle of SVM:**"):
        st.markdown(
        """
        The core idea of SVM is to maximize the margin, which is the distance between the hyperplane and the nearest data points. The margin is defined as:\n
        """)
        st.latex(r"Margin= \frac{2}{∥w∥}")
        st.markdown(
        """
         where ∥w∥ is the Euclidean norm of w.\n
        """)
    # Explanation of the SVM formula (Note: This is a placeholder formula; SVMs use a different formula.)

    # Data Generation section
    st.subheader("Data Generation")
    with st.expander("**How we get random data:**"):
        st.code("np.random.seed(int(seed))")
        st.markdown(
        """
        Purpose:\n
        Sets the seed for the random number generator.\n
        Explanation:\n
        By setting a seed, the random numbers generated will be reproducible. This means that every time you run the code with the same seed, you will get the same sequence of random numbers. This is useful for debugging and ensuring consistent results.\n
        """)
        st.code("X, y = make_blobs(n_samples=100, centers=2, random_state=intseed, cluster_std=1.05)")
        st.markdown(
        """
        Function: make_blobs \n
        This function generates synthetic data points that form distinct clusters. It is commonly used for creating datasets for clustering or classification tasks.\n
        Explanation:\n
        n_samples=100: Specifies the total number of data points to generate. In this case, 100 samples are created.\n
        centers=2: Indicates the number of cluster centers. Here, the data will form 2 distinct clusters, which is suitable for a binary classification problem.\n
        random_state=intseed: Specifies the seed for the random number generator. This ensures reproducibility by setting a fixed seed for the random splitting process.\n
        cluster_std=1.05: Specifies the standard deviation of each cluster. Here, the standard deviation is set to 1.05 to ensure that the clusters are well separated.\n
        Output:\n
        X: A 2D array where each row corresponds to a data point and each column corresponds to a feature. In this case, X will have 100 rows (samples) and 2 columns (features) by default.\n
        y: A 1D array where each element corresponds to the cluster label of the corresponding data point. In this case, y will have 100 elements.\n
        """)
    seed = st.slider("Choose a random seed", 0, 100, 1)
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
    with st.expander("**What we have done:**"):
       st.markdown(
       """
       #### Splitting the Data:\n
       """)
       st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
       st.markdown(
       """
       The dataset is divided into training and testing sets using the train_test_split function.\n
       X and y are the feature and target arrays, respectively.\n
       test_size=0.2 means that 20% of the data is used for testing, while the remaining 80% is used for training.\n
       random_state=42 ensures reproducibility by setting a fixed seed for the random splitting process.\n
       #### Training the Model:\n
           model = SVC(kernel='linear')\n
           model.fit(X_train, y_train)\n
       A SVC model is trained on the training data..\n
       """)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    with st.expander("**What is Accuracy:**"):
        st.markdown(
        """
        #### Definition:\n 
        Accuracy is a measure of how often the model makes the correct prediction. It is the ratio of the number of correct predictions to the total number of predictions made.\n
        #### Formula: 
        If TP is the number of true positives, TN is the number of true negatives, FP is the number of false positives, and FN is the number of false negatives, then accuracy is calculated as:\n
        """)
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.markdown(
        """
        Explanation: In the case, Accuracy:0.XX  means that the model has an accuracy of XX%, indicating that the account of predictions made by the model are correct.
        """)
    st.write("Accuracy:", accuracy)
    # Evaluate the model using accuracy

    # Confusion Matrix section
    with st.expander("**What is Confusion Matrix:**"):
        st.latex(r"\begin{array}{c|cc}& \text{Predicted Positive} & \text{Predicted Negative} \\\hline\text{Actual Positive} & TP & FN \\\text{Actual Negative} & FP & TN \\\end{array}")
        st.markdown(
        """
        #### Definition:\n
        A confusion matrix is a table used to evaluate the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.\n
        #### Structure:\n
        True Positives (TP): The number of instances that were correctly predicted as positive.\n
        True Negatives (TN): The number of instances that were correctly predicted as negative.\n
        False Positives (FP): The number of instances that were incorrectly predicted as positive (Type I error).\n
        False Negatives (FN): The number of instances that were incorrectly predicted as negative (Type II error).\n
        """)
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
    with st.expander("**What is Decision Boundary:**"):
        st.markdown(
        """
        #### Definition of Decision Boundary:\n
        A decision boundary is a boundary line or hyper-surface in a classification problem that separates different classes. It divides the feature space into multiple regions, each corresponding to a specific class. The model uses the decision boundary to determine the class of a sample based on its position in the feature space.\n
        In a geometric sense, the decision boundary is a hyper-surface. For linear classification problems, the decision boundary is a hyper-plane. For example, in a two-dimensional feature space, the decision boundary is a straight line; in a three-dimensional space, it is a plane.\n
        #### Decision Boundaries in Different Supervised Learning Algorithms:\n
        Although the essence of the decision boundary is to separate different classes, its shape and properties can vary across different supervised learning algorithms:\n
        #### Linear Models:\n
        Linear Support Vector Machine (SVM): It seeks an optimal hyper-plane that maximizes the margin between data points of different classes.\n
        """)
    plot_decision_boundary(X_train, y_train, model)
    # Visualize the decision boundary of the SVM model

    # Interactive Prediction section
    st.subheader("Interactive Prediction")
    with st.expander("**PURPOSE:**"):
        st.markdown(
        """
        Just use the model to predict a X value and get the corresponding y value.
        """)
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

    # Part 3
    st.subheader("Do you finish it?")
    if st.button("Yes"):
        st.session_state.part3_visited = True


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
    with st.expander("**Principle of Random Forest:**"):
        st.markdown(
        """
        Random Forests do not have a single, unified formula like linear regression or logistic regression. Instead, they are an ensemble learning method that combines multiple decision trees to make predictions. Here’s how it works conceptually:\n
        #### 1.Decision Trees:\n
        A Random Forest is composed of many individual decision trees. Each tree is trained on a random subset of the data (a process called bootstrapping).\n
        Each tree also considers a random subset of features at each split (a process called feature bagging).\n
        #### 2.Bootstrapping:\n
        For each tree, a random sample of the training data (with replacement) is selected. This means that each tree sees a slightly different version of the data.\n
        #### 3.Feature Bagging:\n
        At each split in the tree, only a random subset of features is considered. This helps to reduce the correlation between trees and improve the model’s robustness.\n
        #### 4.Aggregation:
        For classification tasks, the predictions from all the trees are aggregated by majority voting. The class that gets the most votes from the trees is the final prediction.\n
        For regression tasks, the predictions from all the trees are averaged to produce the final prediction.\n
        """)
    # Data Generation section
    st.subheader("Data Generation")
    with st.expander("**How we get random data:**"):
        st.code("np.random.seed(int(seed))")
        st.markdown(
        """
        Purpose:\n
        Sets the seed for the random number generator.\n
        Explanation:\n
        By setting a seed, the random numbers generated will be reproducible. This means that every time you run the code with the same seed, you will get the same sequence of random numbers. This is useful for debugging and ensuring consistent results.\n
        """)
        st.code("X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=2,random_state=intseed, shuffle=False)\n")
        st.markdown(
        """
        Purpose: \n
        Generates random feature values for the independent variable X.\n
        Explanation:\n
        n_samples=1000:Specifies the total number of data points to generate. In this case, 1000 samples will be created.\n
        n_features=4:Specifies the total number of features (independent variables) for each data point. Here, each sample will have 4 features.\n
        n_informative=2:Specifies the number of features that are informative, meaning they contribute to the classification. In this case, 2 out of the 4 features will be informative and will have a direct impact on the target variable.\n
        n_redundant=2:Specifies the number of redundant features. These features are linear combinations of the informative features. Here, 2 features will be redundant, meaning they are derived from the informative features but add some complexity to the dataset.\n
        n_random_state=intseed:Specifies the seed for the random number generator. This ensures reproducibility by setting a fixed seed for the random splitting process.\n
        shuffle=False:Specifies whether or not to shuffle the data before splitting it into training and testing sets. Here, the data will not be shuffled. This is useful for reproducibility.\n
        """)
    seed = st.slider("Choose a random seed", 0, 100, 1)
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
    with st.expander("**Feature Importance in Random Forest:**"):
        st.markdown(
        """
        Feature Importance is a critical concept in machine learning, especially when using ensemble methods like Random Forest. It helps in understanding which features contribute the most to the prediction of the target variable. In Random Forest, feature importance is typically calculated using one of the following methods:\n
        #### 1. Gini Importance (Mean Decrease in Impurity)\n
        Gini Importance, also known as Mean Decrease in Impurity (MDI), measures the total reduction in node impurity (such as Gini impurity or entropy) brought by a feature across all trees in the forest. The steps to calculate Gini Importance are:\n
        Compute Node Impurity: Calculate the decrease in impurity for each feature when it is used to split a node.\n
        Weight by Node Probability: Multiply the reduction in impurity by the probability of reaching that node.\n
        Sum Across Trees: Sum these values across all trees in the forest for each feature.\n
        However, Gini Importance can be biased towards high cardinality features (features with many unique values) because they tend to create more splits and thus appear more important.\n
        #### 2. Permutation Importance\n
        Permutation Importance evaluates the impact of each feature on the model's performance by randomly shuffling the values of a single feature and measuring the resulting decrease in accuracy. The steps are:\n
        Train Model: Train the Random Forest on the original dataset and observe the accuracy.\n
        Permute Feature Values: Randomly permute the values of a single feature.\n
        Measure Performance Drop: Test the model on the permuted dataset and measure the performance drop.\n
        Repeat: Repeat the process for all features and average the results.\n
        This method is more computationally expensive but provides a more accurate measure of feature importance, especially in the presence of correlated features.\n
        #### 3. SHAP Values (SHapley Additive exPlanations)\n
        SHAP Values are based on cooperative game theory and provide a unified measure of feature importance by explaining the contribution of each feature to individual predictions. The steps to calculate SHAP Values are:\n
        Coalitional Game Theory: Treat all features as players in a cooperative game where the goal is to predict the target variable.\n
        Shapley Values: Calculate the marginal contribution of each feature across all possible subsets of features.\n
        Aggregate: Aggregate these contributions to get the overall feature importance.\n
        SHAP Values offer a comprehensive understanding of feature importance across various data points and are particularly useful for interpreting complex models.\n
        #### Why Feature Importance Matters\n
        Understanding feature importance in Random Forest models is important for several reasons:\n
        Feature Selection: Identifying the most relevant features can improve model performance and reduce overfitting.\n
        Model Interpretability: Knowing which features are most important aids in interpreting the model's behavior and explaining its decisions to stakeholders.\n
        Enhancing Domain Knowledge: Analyzing feature importance can provide insights into the underlying patterns in the data and contribute to domain knowledge.\n
        In summary, feature importance in Random Forest models helps in identifying the most influential features, improving model performance, and providing interpretability. Methods like Gini Importance, Permutation Importance, and SHAP Values each offer unique insights into feature contributions\n
        """)
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
    with st.expander("**What is Confusion Matrix:**"):
        st.latex(r"\begin{array}{c|cc}& \text{Predicted Positive} & \text{Predicted Negative} \\\hline\text{Actual Positive} & TP & FN \\\text{Actual Negative} & FP & TN \\\end{array}")
        st.markdown(
        """
        #### Definition:\n
        A confusion matrix is a table used to evaluate the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.\n
        #### Structure:\n
        True Positives (TP): The number of instances that were correctly predicted as positive.\n
        True Negatives (TN): The number of instances that were correctly predicted as negative.\n
        False Positives (FP): The number of instances that were incorrectly predicted as positive (Type I error).\n
        False Negatives (FN): The number of instances that were incorrectly predicted as negative (Type II error).\n
        """)
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    # Evaluate the model using a confusion matrix

    # Decision Boundary section
    st.subheader("Decision Boundary")
    with st.expander("**What is Decision Boundary:**"):
        st.markdown(
        """
        #### Definition of Decision Boundary:\n
        A decision boundary is a boundary line or hyper-surface in a classification problem that separates different classes. It divides the feature space into multiple regions, each corresponding to a specific class. The model uses the decision boundary to determine the class of a sample based on its position in the feature space.\n
        In a geometric sense, the decision boundary is a hyper-surface. For linear classification problems, the decision boundary is a hyper-plane. For example, in a two-dimensional feature space, the decision boundary is a straight line; in a three-dimensional space, it is a plane.\n
        #### Decision Boundaries in Different Supervised Learning Algorithms:\n
        Although the essence of the decision boundary is to separate different classes, its shape and properties can vary across different supervised learning algorithms:\n
        #### Linear Models:\n
        Logistic Regression: The decision boundary is linear, defined by a hyper-plane that divides the feature space into two classes. For example, in a two-dimensional space, the decision boundary is a straight line.\n
        Linear Support Vector Machine (SVM): It seeks an optimal hyper-plane that maximizes the margin between data points of different classes.\n
        #### Non-linear Models
        Polynomial Regression: By introducing polynomial features, the decision boundary can become non-linear.\n
        K-Nearest Neighbors (KNN): The decision boundary is determined by the local neighborhood of training data points and is usually complex and discontinuous.\n
        Decision Trees: The decision boundary is formed by a series of rules (typically axis-parallel) and appears as a step-like or zigzag shape.\n
        Neural Networks: They can learn complex non-linear decision boundaries, the shape of which depends on the network structure and training process.\n
        #### Summary\n
        The shape and complexity of the decision boundary depend on the classification algorithm used and the distribution of the data. Linear models typically produce simple linear boundaries, while non-linear models can adapt to more complex data distributions and produce non-linear boundaries. Understanding the decision boundaries of different algorithms helps in selecting the appropriate model and optimizing classification performance.\n
        """)
    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, rf, alpha=0.4, cmap='coolwarm')
    # Visualize the decision boundary of the Random Forest model

    # Part 4
    st.subheader("Do you finish it?")
    if st.button("Yes"):
        st.session_state.part4_visited = True

        
        
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
        st.error("Incorrect. Have another guess!")
    
    
    
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
        st.error("Incorrect. Have another guess!")
        
        
    
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
        st.error("Incorrect. Have another guess!")
    
    
    
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
        st.error("Incorrect. Have another guess!")
        