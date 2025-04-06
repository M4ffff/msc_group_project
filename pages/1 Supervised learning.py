import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import make_classification,make_blobs
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


from Modules.supervised_functions import subpage1, subpage2, subpage3, subpage4, supervised_quiz


st.title("Supervised Learning ")

tab1, tab2, tab3, tab4 = st.tabs(['Introduction', 'Different models', 'Car Accident Prediction', 'Film Rating Prediction'])


with tab1:

    st.write("🎉🎉🎉If you've made it this far, it means you're genuinely interested in this topic, and the content ahead is definitely worth looking forward to!🎉🎉🎉")

    st.header("What is Supervised Learning ")
    st.write("Supervised learning is one of the core techniques in machine learning. Think of it like teaching a model with examples: you provide it with data (the inputs) along with the correct answers (the outputs), and it learns to make predictions based on that. These predictions can be continuous values (like predicting house prices—this is called regression) or categories (like identifying if an email is spam or not—this is called classification).")
    st.write("After being trained on labeled examples, the model can then make educated guesses on new, unseen data. This ability to generalize is key, and it mirrors how humans learn to recognize patterns and form concepts.")

    st.header("Supervised Learning in the ***:rainbow[real-world]***")
    st.write("Due to its powerful predictive capabilities and broad applicability, supervised learning has a wide range of applications in many fields. Here are some of the main application areas:")
    
    st.subheader("Image Recognition")
    st.write("Facial Recognition: Used in security systems and social media (e.g., for automatic photo tagging).")
    st.write("Object Detection: Detecting pedestrians and vehicles in autonomous driving, and abnormal behaviors in security surveillance.")
    st.write("Medical Imaging Analysis: Used for disease diagnosis (e.g., cancer detection, X-ray image analysis).")

    #Make image in the middle of screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/1_supervised_images/Facial Recognition.jpg", caption="Facial Recognition", use_container_width=True)

    st.subheader("Natural Language Processing (NLP)")
    st.write("Sentiment Analysis: Analyzing the sentiment (positive, negative, or neutral) in text (e.g., social media posts, product reviews).")
    st.write("Machine Translation: Translating text from one language to another.")
    st.write("Converting speech signals into text (e.g., in smart voice assistants)")
    st.write("Text Classification: Categorizing text into different classes (e.g., news classification, spam detection).")

    # Make image in the middle of screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/1_supervised_images/ORC.png", caption="ORC", use_container_width=True)

    st.subheader("Gaming and Entertainment")
    st.write("Game Recommendation: Recommending games based on players' gaming history and preferences.")
    st.write("In-Game Character Behavior Prediction: Predicting character behavior based on player data to optimize the gaming experience.")

    # Make image in the middle of screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/1_supervised_images/OIP.jpg", caption="OIP", use_container_width=True)

    
    st.write("Of course, there are many other application scenarios. Supervised learning is widely applied and deeply integrated into various aspects of daily life, making it an accessible and practical tool.")

with tab2:
    # Initialize session state
    if "part1_visited" not in st.session_state:
        st.session_state.part1_visited = False
    if "part2_visited" not in st.session_state:
        st.session_state.part2_visited = False
    if "part3_visited" not in st.session_state:
        st.session_state.part3_visited = False
    if "part4_visited" not in st.session_state:
        st.session_state.part4_visited = False
    st.header("Which Supervised Learning model would you like?")
    st.write("No matter where it is applied, the underlying logic is consistent. Starting from the basics is a necessary path to becoming a master. Choose a direction that interests you and dive in. :diving_mask: ")

    #
    summaries = {
        "Linear Regression": "Linear regression is one of the most fundamental supervised learning algorithms, primarily used for predicting continuous target variables. It assumes a linear relationship between input features and the target variable and attempts to find the best-fitting line to describe this relationship.",
        "Logistic Regression": "Logistic regression is mainly used for binary classification problems. It maps the results of linear regression to values between 0 and 1 using a logistic function, thereby predicting the probability of an event occurring.",
        "Support Vector Machine": "Support Vector Machine (SVM) is a powerful classification algorithm that maximizes the margin between different classes by finding an optimal decision boundary. SVM can also be used for regression problems.",
        "Random Forest": "Random Forest is an ensemble learning method that improves the generalization ability of the model by constructing multiple decision trees and averaging or voting on their results."
    }

    #
    selected_method = st.selectbox(
        "Pick a model!",
        options=list(summaries.keys())
    )

    #
    if selected_method == "Linear Regression":
        subpage1(summaries[selected_method])
    elif selected_method == "Logistic Regression":
        subpage2(summaries[selected_method])
    elif selected_method == "Support Vector Machine":
        subpage3(summaries[selected_method])
    elif selected_method == "Random Forest":
        subpage4(summaries[selected_method])

    st.divider()
    parts = {
        "part1_visited": "Part 1:Linear Regression ",
        "part2_visited": "Part 2: Logistic Regression",
        "part3_visited": "Part 3: SVM",
        "part4_visited": "Part 4: Random Forest"
    }
    incomplete_parts = [
        name for key, name in parts.items()
        if not st.session_state.get(key, False)
    ]
    #show quizz
    if not incomplete_parts:
        st.subheader("Quiz time!")
        st.write("How much have you learnt?!?")
        supervised_quiz()
    else:
        for i in range(1, 5):
            status = "✅" if st.session_state.get(f"part{i}_visited", False) else "❌"
            st.write(f"{status} Part {i}")
        st.warning(
            "Please complete the following parts before taking the quiz:\n\n" +
            "\n".join(f"{name}\n" for name in incomplete_parts )
        )


with tab3:
    
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Here , we created a model using **real data** to classify the severity of car crashes. We're using a Random Forest classifier, which is a powerful machine learning method based on ensembles of decision trees. The goal is to understand which conditions are predictive of crash severity.")
    
    with col2:
        st.image('images/1_supervised_images/cars_onroad.jpg')
    
    # Load Dataset
    @st.cache_data
    def clean_data(df):
        
        # st.dataframe(df.head())

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

    data = pd.read_csv('datasets/Collisions.csv')
    
    # Sample Features
    df = data[['SEVERITYCODE', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'SPEEDING']] # +junctiontype/ addrtype

    st.write("This is the data we're using:")
    st.dataframe(df.head())
    
    st.write("Some of the columns contain **categorial values** (e.g ., weather conditions), which machine learning models can't directly process. To get around this, we need to convert them to numbers using an encoder")
    st.write("There are also missing vlaues which need to be cleaned. For example, missing or blank entries in the 'SPEEDING' column are replaced with 'N' (No), and then encoded as 0 or 1.")
    
    # Load data
    df = clean_data(df)
    
    st.dataframe(df.head())


    df.fillna('0', inplace=True) 

    X = df.drop(columns=['SEVERITYCODE'])
    y = df['SEVERITYCODE']

    st.header('Predicting Accident Severity - Random Forest Classifier')

    st.write("Lets look at how crash severity values are distributed:")

    # Visualize Class Distribution of Severity Code
    st.subheader(" Distribution of Accident Severity")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x=df['SEVERITYCODE'], palette='viridis', ax=ax)
    ax.set_xlabel("Severity Code")
    ax.set_ylabel("Count")
    # ax.set_yscale('log')
    st.pyplot(fig)

    st.write("The plot above shows the severity distribution of car accidents in Seattle:")
    st.markdown("""
    - **0** = Unknown  
    - **1** = Property Damage Only  
    - **2** = Minor Injury  
    - **3** = Serious Injury  
    - **4** = Fatality  

    Most accidents result in property damage or minor injuries, while fatal crashes are rare.
    """)

    # Streamlit Sidebar for Feature Selection
    st.header("Select Features for Prediction")
    
    st.write("Now we’ll let you choose which features to use to train the model. This helps us understand which factors actually contribute to predicting severity.")

    selected_features = st.multiselect("Choose features to include", X.columns.tolist(), default=X.columns.tolist())

    # Train Model on Selected Features
    if selected_features:
        if X[selected_features].shape[0] == 0:
            st.error("No data available after filtering. Try selecting different features.")
            st.stop()
       
    
        model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=1)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        perc_accuracy = accuracy * 100

        st.write("## Model Performance")
        
        # Display Performance Metrics
        st.write("The model was trained using the selected features. Below is the overall accuracy on unseen data:")
        
        st.write(f"**Model Accuracy with Selected Features:** {accuracy:.4f}")
        st.write(f"This means that the model correctly classified {perc_accuracy:.4f}% of accident cases")
        
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        })

        if feature_importances["Importance"].sum() == 0:
            st.error("All feature importances are zero. Try selecting different features or check dataset preprocessing.")
            st.stop()

        # Feature Importance Visualization
        st.write("### Feature Importance")
        feature_importances = pd.DataFrame({'Feature': selected_features, 'Importance': model.feature_importances_}).sort_values(by="Importance", ascending=False)

        st.write("This shows how much each feature contributed to the model's predictions.")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Importance", y="Feature", data=feature_importances, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("Please select at least one feature to train the model.")

    st.write("This helps us understand which factors are most influential in determining accident severity. For example, a feature with high importance means it's a strong indicator of crash severity.")
    
    st.write("By analyzing this, we can make data-informed decisions about road safety measures, enforcement priorities, and public awareness.")


with tab4:
    @st.cache_data
    def load_imdb_data():
        df = pd.read_csv('datasets/IMDb_Dataset.csv')

        # Encode 'Certificates' and 'Genre'
        encoder = LabelEncoder()
        df['Certificates'] = encoder.fit_transform(df['Certificates'])
        df['Genre'] = encoder.fit_transform(df['Genre'])

        df = df[['IMDb Rating', 'Certificates', 'Duration (minutes)', 'Year', 'MetaScore', 'Genre']]
        df = df.dropna()

        return df

    # Load data
    df_imdb = load_imdb_data()

    # Split features and target
    X_imdb = df_imdb.drop(columns=['IMDb Rating'])
    y_imdb = df_imdb['IMDb Rating']

    st.header("Predicting IMDb Film Ratings with Random Forest Regression")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        In this section, we're using **real movie data** from IMDb to build a regression model that predicts a film’s rating out of 10.
        Since ratings are continuous values (not categories), we use a **regression model**—specifically, Random Forest Regression.

        This allows us to explore how features like **film duration**, **release year**, and **age certificate** relate to the overall score.
        """)
    with col2:
        st.image('images/1_supervised_images/imdb.png')

    st.subheader("Data Preview")
    st.write("Here’s a preview of the dataset used for training:")
    st.dataframe(df_imdb.head())

    st.subheader("Select Features for Prediction")
    st.write("Choose which features to include in the model. This helps us explore their influence on the final rating.")
    selected_features_imdb = st.multiselect("Choose features", X_imdb.columns.tolist(), default=X_imdb.columns.tolist())


    if selected_features_imdb:
        X_train, X_test, y_train, y_test = train_test_split(X_imdb[selected_features_imdb], y_imdb, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance")
        st.write("""
        These values tell us how well the model is performing:
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings.
        - **MSE (Mean Squared Error)**: Penalizes larger errors more heavily.
        - **R² Score**: Indicates how much variance in the rating is explained by the model (closer to 1 is better).
        """)
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Scatter plot: actual vs predicted
        st.subheader("Actual vs Predicted Ratings")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='teal')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
        ax.set_xlabel("Actual IMDb Rating")
        ax.set_ylabel("Predicted IMDb Rating")
        ax.set_title("Actual vs Predicted Ratings")
        st.pyplot(fig)

        # Coefficients
        coefficients = pd.DataFrame({
            'Feature': selected_features_imdb,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", key=abs, ascending=False)

        st.subheader("Feature Importance")
        st.write("""
        These coefficients indicate how much each feature affects the predicted rating. 
        A **positive value** means the feature increases the predicted score, while a **negative value** means it reduces it.
        Larger absolute values indicate more influence.
        """)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Importance", y="Feature", data=coefficients, ax=ax)
        ax.set_title("Feature importance (RF Regression)")
        st.pyplot(fig)

        st.write("For example, a longer movie might have a slightly higher predicted rating, or older movies may tend to score better.")

        # Quiz
        st.header("Quiz: Identify the Least Important Feature!")

        if not coefficients.empty:
            least_important_feature = coefficients.iloc[-1]['Feature']
            quiz_options = coefficients['Feature'].tolist()

            selected_answer = st.radio(
                "Which feature has the **least** impact (smallest importance) in the model?",
                quiz_options, index=None
            )

            if st.button("Submit Answer 2"):
                if selected_answer == least_important_feature:
                    st.success(f"✅ Correct! The least important feature is **{least_important_feature}**.")
                elif selected_answer is None:
                    st.write("")
                else:
                    st.error(f"❌ Incorrect. Have another go!")
    else:
        st.warning("Please select at least one feature to train the model.")
    