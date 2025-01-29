import streamlit as st
import numpy as np
import pandas as pd
import time

st.title("Supervised Learning ")
st.write("🎉🎉🎉If you've made it this far, it means you're genuinely interested in this topic, and the content ahead is definitely worth looking forward to!🎉🎉🎉")

st.header("1.What is Supervised Learning ")
st.write("&nbsp;&nbsp;&nbsp;&nbsp;Supervised learning,also known as supervised learning or supervised learning,is a method in machine learning that can learn or establish a pattern (function/learning model) from training data and use this pattern to infer new instances.The training data consists of input objects (usually vectors) and expected outputs. The output of the function can be a continuous value (referred to as regression analysis) or the prediction of a classification label (known as classification).")
st.write("&nbsp;&nbsp;&nbsp;&nbsp;The task of a supervised learner is to predict the output of this function for any possible input after observing some pre-labeled training examples (inputs and expected outputs). To achieve this, the learner must generalize from the existing data to unobserved cases in a reasonable manner (see inductive bias). In human and animal perception, this is commonly referred to as concept learning.")
st.write("Description from [Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning)")

st.header("2.What is Supervised Learning ")
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

st.write("&nbsp;&nbsp;&nbsp;&nbsp;Of course, there are many other application scenarios.Supervised learning is widely applied and deeply integrated into various aspects of daily life, making it an accessible and practical tool rather than something distant or unattainable.")

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

# 根据选择显示简要概述
st.header("Introduction")
st.write(summaries[selected_method])




