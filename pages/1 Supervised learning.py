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
from sklearn.datasets import make_classification,make_blobs
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

tab1, tab2 = st.tabs(['Introduction', 'Car Accident Prediction'])

with tab1:

    # def about subpage
    def subpage1():
        st.write(summaries[selected_method])
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

    def subpage2():
        st.write(summaries[selected_method])

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

        def plot_decision_boundary(X, y, model):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Decision Boundary')
            plt.show()

        plot_decision_boundary(X_train, y_train, model)

        st.subheader("Interactive Prediction")
        feature1 = st.slider("Choose a value for Feature 1", X[:, 0].min(), X[:, 0].max(), (X[:, 0].mean()))
        feature2 = st.slider("Choose a value for Feature 2", X[:, 1].min(), X[:, 1].max(), (X[:, 1].mean()))
        input_features = np.array([[feature1, feature2]])
        prediction = model.predict(input_features)
        st.write(f"Predicted class for input ({feature1:.2f}, {feature2:.2f}): {prediction[0]}")

    def subpage3():
        st.write(summaries[selected_method])

        st.subheader("Formula")
        st.latex(r"f(x) = \text{sign}(w \cdot x + b)")

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

        def plot_decision_boundary(X, y, model):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Decision Boundary')
            plt.show()

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

        plot_decision_boundary_with_hyperplane(X_train, y_train, model)

    def subpage4():
        st.write(summaries[selected_method])
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

        def plot_decision_boundary(X, y, model):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm', alpha=0.8)
            plt.title('Decision Boundary')
            st.pyplot(plt)

        plot_decision_boundary(X_train, y_train, rf)


    st.title("Supervised Learning ")
    st.write("🎉🎉🎉If you've made it this far, it means you're genuinely interested in this topic, and the content ahead is definitely worth looking forward to!🎉🎉🎉")

    st.header("1. What is Supervised Learning ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Supervised learning, also known as supervised learning or supervised learning, is a method in machine learning that can learn or establish a pattern (function/learning model) from training data and use this pattern to infer new instances. The training data consists of input objects (usually vectors) and expected outputs. The output of the function can be a continuous value (referred to as regression analysis) or the prediction of a classification label (known as classification).")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;The task of a supervised learner is to predict the output of this function for any possible input after observing some pre-labeled training examples (inputs and expected outputs). To achieve this, the learner must generalize from the existing data to unobserved cases in a reasonable manner (see inductive bias). In human and animal perception, this is commonly referred to as concept learning.")
    st.write("Description from [Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning)")

    st.header("2. What is Supervised Learning ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Supervised learning is one of the most common methods in machine learning. It trains models using labeled training data, enabling the models to make predictions or classifications on new, unseen data. ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Due to its powerful predictive capabilities and broad applicability, supervised learning has a wide range of applications in many fields. Here are some of the main application areas:")
    st.subheader("&nbsp;&nbsp;1. Image Recognition")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Facial Recognition: Used in security systems and social media (e.g., for automatic photo tagging).")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Object Detection: Detecting pedestrians and vehicles in autonomous driving, and abnormal behaviors in security surveillance.")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;Medical Imaging Analysis: Used for disease diagnosis (e.g., cancer detection, X-ray image analysis).")
    col1, col2, col3 = st.columns([1, 2, 3])
    with col2:
        st.image("images/Facial Recognition.jpg", caption="Facial Recognition", width=300)

        st.subheader("&nbsp;&nbsp;2. Natural Language Processing (NLP)")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Sentiment Analysis: Analyzing the sentiment (positive, negative, or neutral) in text (e.g., social media posts, product reviews).")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Machine Translation: Translating text from one language to another.")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Converting speech signals into text (e.g., in smart voice assistants)")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Text Classification: Categorizing text into different classes (e.g., news classification, spam detection).")
    col1, col2, col3 = st.columns([1, 2, 3])
    with col2:
        st.image("images/ORC.png", caption="ORC", width=300)

        st.subheader("&nbsp;&nbsp;3.Gaming and Entertainment")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Game Recommendation: Recommending games based on players' gaming history and preferences.")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;In-Game Character Behavior Prediction: Predicting character behavior based on player data to optimize the gaming experience.")
        col1, col2, col3 = st.columns([1, 2, 3])
    with col2:
        st.image("images/OIP.jpg", caption="OIP", width=300)

        st.write("&nbsp;&nbsp;&nbsp;&nbsp;Of course, there are many other application scenarios. Supervised learning is widely applied and deeply integrated into various aspects of daily life, making it an accessible and practical tool rather than something distant or unattainable.")

        st.header("3.What kind of Supervised Learning do you like")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;But no matter where it is applied, the underlying logic is consistent. Starting from the basics is a necessary path to becoming a master. Choose a direction that interests you and dive in.")

    summaries = {
        "Linear Regression": "&nbsp;&nbsp;&nbsp;&nbsp;Linear regression is one of the most fundamental supervised learning algorithms, primarily used for predicting continuous target variables. It assumes a linear relationship between input features and the target variable and attempts to find the best-fitting line to describe this relationship.",
        "Logistic Regression": "&nbsp;&nbsp;&nbsp;&nbsp;Logistic regression is mainly used for binary classification problems. It maps the results of linear regression to values between 0 and 1 using a logistic function, thereby predicting the probability of an event occurring.",
        "Support Vector Machine": "&nbsp;&nbsp;&nbsp;&nbsp;Support Vector Machine (SVM) is a powerful classification algorithm that maximizes the margin between different classes by finding an optimal decision boundary. SVM can also be used for regression problems.",
        "Random Forest": "&nbsp;&nbsp;&nbsp;&nbsp;Random Forest is an ensemble learning method that improves the generalization ability of the model by constructing multiple decision trees and averaging or voting on their results."
    }
    selected_method = st.selectbox(
        "Choose an option that you like",
        options=list(summaries.keys())
    )

    if selected_method == "Linear Regression":
        subpage1()
    elif selected_method == "Logistic Regression":
        subpage2()
    elif selected_method == "Support Vector Machine":
        subpage3()
    elif selected_method == "Random Forest":
        subpage4()

with tab2:
    # Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/Collisions.csv')
    df = df[['SEVERITYCODE', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'SPEEDING']]  # Sample Features

    df['SPEEDING'] = df['SPEEDING'].fillna('N')
    df['SPEEDING'] = df['SPEEDING'].replace(r'^\s*$', 'N', regex=True)

    df['SPEEDING'] = df['SPEEDING'].map({'Y': 1, 'N': 0})
    df = df.dropna()  # Drop missing values

    df['SEVERITYCODE'] = df['SEVERITYCODE'].map({'0': 0, '1':1, '2':2, '2b':3, '3':4})  
    
    # List of categorical features
    categorical_features = ['WEATHER', 'ROADCOND', 'LIGHTCOND']

    # Convert categorical features to numerical labels
    encoder = OrdinalEncoder()
    df[categorical_features] = encoder.fit_transform(df[categorical_features])
    return df

# Load data
df = load_data()

df.fillna('0', inplace=True) 

X = df.drop(columns=['SEVERITYCODE'])
y = df['SEVERITYCODE']

st.header('Predicting Accident Severity - Random Forest Classifier')

# Visualize Class Distribution of Severity Code
st.subheader(" Distribution of Accident Severity")
fig, ax = plt.subplots()
sns.countplot(x=df['SEVERITYCODE'], palette='viridis', ax=ax)
ax.set_xlabel("Severity Code")
ax.set_ylabel("Count")
ax.set_yscale('log')
st.pyplot(fig)

st.write("Dataset Preview:", df.head())  # Show first 5 rows

st.image('images/cars_onroad.jpg')

# Streamlit Sidebar for Feature Selection
st.header("Select Features for Prediction")
selected_features = st.multiselect("Choose features to include", X.columns.tolist(), default=X.columns.tolist())

# Train Model on Selected Features
if selected_features:
    if X[selected_features].shape[0] == 0:
        st.error("No data available after filtering. Try selecting different features.")
        st.stop()

    # Sliders for Model Parameters (Main Layout)
    st.header("🔧 Adjust Model Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)

    with col2:
        max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)

    with col3:
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)        
   
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.write("Feature Importances:", model.feature_importances_)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display Performance Metrics
    st.write("## Model Performance")
    st.write(f"**Model Accuracy with Selected Features:** {accuracy:.4f}")
    
    feature_importances = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    })

    if feature_importances["Importance"].sum() == 0:
        st.error("All feature importances are zero. Try selecting different features or check dataset preprocessing.")
        st.stop()

    # Feature Importance Visualization
    st.write("**Feature Importance**")
    feature_importances = pd.DataFrame({'Feature': selected_features, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=feature_importances, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
else:
    st.warning("Please select at least one feature to train the model.")