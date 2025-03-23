import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

# Page and side bar title.
st.markdown("# The Basics of Machine Learning")


# Subtitle for brief intro to machine learning.
st.markdown("**Welcome to a beginners guide to Machine Learning techniques and their applications!**")

# Brief intro paragraph.
st.markdown("Machine learning has become an integral part of solving ***real-world*** problems, making significant strides across diverse domains like healthcare:hospital:, finance:money_with_wings:, transportation:car:, and entertainment:movie_camera:. This web app is designed to serve as an accessible guide for beginners, helping you understand and apply key machine learning techniques to ***real-world*** datasets.")


# Following intro paragraph (draws attention to real-world applications).
st.markdown("Whether you’re curious about how Netflix recommends your favourite shows or how self-driving cars navigate roads, this app will provide the foundational knowledge you need to explore these innovations further.")


# Introduction to content.
# Add slider to show how ML techniques work e.g. for linear regression show how changing the slope or intercept affects a line of best fit on a graph.
st.subheader("What will you learn?")
st.markdown("This web app is structured into several sections, each focusing on different essential machine learning concepts.")

with st.expander("***:rainbow[Section One:]*** *Supervised Learning*"):
    st.markdown('''
    - Linear Regression
    
    - Logistic Regression
    
    - Support Vector Machines (SVMs)
    
    - Random Forests''')


with st.expander("***:rainbow[Section Two:]*** *Unsupervised Learning*"):
    st.markdown('''
    - Principle component analysis (PCA)
    
    - Clustering''')


with st.expander("***:rainbow[Section Three:]*** *Neural Networks*"):
    st.markdown('''
    - Build your own Neural Network classification tool!''')


# Demystifying machine learning terms.
st.subheader("What's the difference between AI, ML, Deep Learning, and Generative AI?")
st.markdown("-	Artificial Intelligence (AI): The broad field of creating systems capable of intelligent behaviour, from playing chess to making medical diagnoses.")
st.markdown("-	Machine Learning (ML): A subset of AI focused on developing algorithms that enable computers to learn patterns from data and make predictions.")
st.markdown("-	Deep Learning: A specialised branch of ML that uses neural networks with many layers to solve complex problems like image and speech recognition.")
st.markdown("-	Generative AI: A type of AI focused on generating new data, such as text, images, or audio, based on training examples, such as ChatGPT or DALL-E.")

st.image("images/AI-venn-diagram.png", caption="A diagram showing the relationship between AI, ML, deep learning, and generative AI.", use_column_width=True)

# Connecting machine learning to real world examples.
# Add dropdown/selectbox to let users select a real-world application, display short description or visual of how ML is used in that field.
st.subheader("How can ML be applied in the real-world?")
fields = ["Healthcare", "Finance", "Transportation", "Entertainment"]
fields_choice = st.selectbox("**Pick one!**", fields)

if fields_choice == "Healthcare":
    st.write("ML algorithms can help diagnose diseases, analyse medical images, and predict patient outcomes.")
    st.image("images/ML-in-healthcare.jpg", caption=" ", use_column_width=True)

elif fields_choice == "Finance":
    st.write("ML can be used in fraud detection, algorithmic trading, and credit scoring.")
    st.image("images/ML-in-finance.jpg", caption=" ", use_column_width=True)
    
elif fields_choice == "Transportation":
    st.write("ML systems help create autonomous vehicles, route optimisation, and predictive maintenance.")
    st.image("images/ML-in-transportation.png", caption=" ", use_column_width=True)
    
elif fields_choice == "Entertainment":
    st.write("ML algorithms also give us content recommendation systems, generative art, and virtual reality.")
    st.image("images/ML-in-entertainment.jpg", caption=" ", use_column_width=True)


# Brief insight into AI.
st.subheader("Why is AI so important?")
st.markdown("AI and ML are at the forefront of technological advancement, driving innovation in science and industry. From accelerating drug discovery in pharmaceuticals to improving manufacturing efficiency and developing groundbreaking technologies in renewable energy, their impact is profound and far-reaching. By gaining a foundational understanding of ML, you’ll join the growing community of individuals shaping the future of these transformative fields.")


# Quiz.
st.subheader("Quiz time!")
question_one = st.radio(
    "What is meant by the term AI?",
    ("A type of data analysis.", "A type of electrical outlet.", "A system capable of intelligent behaviour."), index=None)
if question_one == "A system capable of intelligent behaviour.":
    st.success("Correct!")
elif question_one == None:
    st.write("")
else:
    st.error("Try again!")

question_two = st.radio(
    "How can ML algorithms benefit Healthcare?",
    ("By giving patients an apple a day to keep the doctor away.", "By analysing medical images.", "By teleporting patients running late to their appointments."), index=None)
if question_two == "By analysing medical images.":
    st.success("Correct!")
elif question_two == None:
    st.write("")
else:
    st.error("Try again!")

question_three = st.radio(
    "Which of these ML techniques does linear regression fall under?",
    ("Unsupervised Learning.", "Superduper Learning.", "Supervised Learning."), index=None)
if question_three == "Supervised Learning.":
    st.success("Correct!")
elif question_three == None:
    st.write("")
else:
    st.error("Try again!")


# Star rating to let users' rate how much they know about ML before starting.
st.markdown("**How much do you know about ML?**")
knowledge_slider = st.slider("1=Beginner to 5=Expert:", 1, 5, 1)
st.write("Your current knowledge level:", knowledge_slider)
st.write("(Keep note of me to reflect after completing the content!)")


# Progress bar to indicate the users learning journey.
st.subheader("Get started!")
st.markdown("Dive into the sections, experiment with datasets, and watch your skills grow as you progress through this app. Let’s demystify machine learning together and unlock its potential to address real-world challenges!")
st.progress(10)