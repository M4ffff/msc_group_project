import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

# Page and side bar title.
st.markdown("# The Basics of Machine Learning")
st.sidebar.markdown("# Content Menu")


# Side bar menu to add links.
st.sidebar.markdown("***Section One: Supervised Learning.***")
st.sidebar.markdown("***Section Two: Unsupervised Learning.***")
st.sidebar.markdown("***Section Three: Neural Networks.***")
st.sidebar.markdown("***Section Four: Python Implementation.***")


# Subtitle for brief intro to machine learning.
st.markdown("**Welcome to a beginners guide to Machine Learning techniques and their applications!**")

# Brief intro paragraph.
st.markdown("Machine learning has become an integral part of solving real-world problems, making significant strides across diverse domains like healthcare, finance, transportation, and entertainment. This web app is designed to serve as an accessible guide for beginners, helping you understand and apply key machine learning techniques to real-world datasets.")


# Following intro paragraph (draws attention to real-world applications).
st.markdown("Whether you’re curious about how Netflix recommends your favourite shows or how self-driving cars navigate roads, this app will provide the foundational knowledge you need to explore these innovations further.")


# Introduction to content.
# Add slider to show how ML techniques work e.g. for linear regression show how changing the slope or intercept affects a line of best fit on a graph.
st.markdown("# What will you learn?")
st.markdown("This web app is structured into several sections, each focusing on different essential machine learning concepts.")

st.markdown("***Section One***: *Supervised Learning:*")
st.markdown("- Linear regression: Predict continuous outcomes like housing prices or stock market trends.")
st.markdown("-	Classification: Categorise data, such as spam email detection or predicting customer churn.")


st.markdown("***Section Two***: *Unsupervised Learning:*")
st.markdown("-	Clustering: Group similar data points, such as a customer segmentation or image compression.")
st.markdown("-	Principle component analysis (PCA): Reduce the dimensionality of datasets, helping visualise and analyse complex data efficiently.")


st.markdown("***Section Three***: *Neural Networks:*")
st.markdown("-	Understand the building blocks of artificial intelligence and how they power applications like fraud detection, speech recognition, and personalised recommendations.")


st.markdown("***Section Four***: *Virtual Reality (VR) Convolutional Neural Networks:*")
st.markdown("-	Visualise CNNs and their functions in 3D space.")


st.markdown("***Section Five***: *Python Implementation:*")
st.markdown("-	This section goes deeper into the principles and theory of each method.")
st.markdown("- Learn Learn how to implement each method in Python with statis code examples, catering to more advanced learners who want hands-on experience with the tools and techniques.")


st.markdown("**Example:** Look at how changing the slope or intercept affects the line of best fit on a graph!")
x = np.linspace(0, 10, 100)
slope = st.slider("Choose slope (m):", 0.1, 5.0, 1.0)
intercept = st.slider("Choose intercept (b):", -5.0, 5.0, 0.0)
y = slope * x + intercept

fig, ax = plt.subplots()
ax.plot(x, y, label=f"y = {slope:.2f}x + {intercept:.2f}")
ax.legend()
ax.set(xlabel="$x-axis$", ylabel="$y-axis$", title="Linear Regression")
ax.set_xlim(-2, 12)
ax.set_ylim(-10, 60)
st.pyplot(fig)


# Demystifying machine learning terms.
st.markdown("# What's the difference between AI, ML, Deep Learning, and Generative AI?")
st.markdown("-	Artificial Intelligence (AI): The broad field of creating systems capable of intelligent behaviour, from playing chess to making medical diagnoses.")
st.markdown("-	Machine Learning (ML): A subset of AI focused on developing algorithms that enable computers to learn patterns from data and make predictions.")
st.markdown("-	Deep Learning: A specialised branch of ML that uses neural networks with many layers to solve complex problems like image and speech recognition.")
st.markdown("-	Generative AI: A type of AI focused on generating new data, such as text, images, or audio, based on training examples, such as ChatGPT or DALL-E.")

st.image("AI-venn-diagram.png", caption="A diagram showing the relationship between AI, ML, deep learning, and generative AI.", use_column_width=True)

# Connecting machine learning to real world examples.
# Add dropdown/selectbox to let users select a real-world application, display short description or visual of how ML is used in that field.
st.markdown("# How can ML be applied in the real-world?")
fields = ["Healthcare", "Finance", "Transportation", "Entertainment"]
fields_choice = st.selectbox("**Pick one!**", fields)

if fields_choice == "Healthcare":
    st.write("ML algorithms can help diagnose diseases, analyse medical images, and predict patient outcomes.")
    st.image("ML-in-healthcare.jpg", caption=" ", use_column_width=True)

elif fields_choice == "Finance":
    st.write("ML can be used in fraud detection, algorithmic trading, and credit scoring.")
    st.image("ML-in-finance.jpg.webp", caption=" ", use_column_width=True)
    
elif fields_choice == "Transportation":
    st.write("ML systems help create autonomous vehicles, route optimisation, and predictive maintenance.")
    st.image("ML-in-transportation.png", caption=" ", use_column_width=True)
    
elif fields_choice == "Entertainment":
    st.write("ML algorithms also give us content recommendation systems, generative art, and virtual reality.")
    st.image("ML-in-entertainment.jpg", caption=" ", use_column_width=True)


# Brief insight into AI.
st.markdown("# Why is AI so important?")
st.markdown("AI and ML are at the forefront of technological advancement, driving innovation in science and industry. From accelerating drug discovery in pharmaceuticals to improving manufacturing efficiency and developing groundbreaking technologies in renewable energy, their impact is profound and far-reaching. By gaining a foundational understanding of ML, you’ll join the growing community of individuals shaping the future of these transformative fields.")


# Quiz.
st.markdown("# Quiz time!")
question_one = st.radio(
    "What is meant by the term AI?",
    ("A type of data analysis.", "A type of electrical outlet.", "A system capable of intelligent behaviour."))
if question_one == "A system capable of intelligent behaviour.":
    st.success("Correct!")
else:
    st.error("Try again!")

question_two = st.radio(
    "How can ML algorithms benefit Healthcare?",
    ("By giving patients an apple a day to keep the doctor away.", "By analysing medical images.", "By teleporting patients running late to their appointments."))
if question_two == "By analysing medical images.":
    st.success("Correct!")
else:
    st.error("Try again!")

question_three = st.radio(
    "Which of these ML models does linear regression fall under?",
    ("Unsupervised Learning.", "Superduper Learning.", "Supervised Learning."))
if question_three == "Supervised Learning.":
    st.success("Correct!")
else:
    st.error("Try again!")


# Star rating to let users' rate how much they know about ML before starting.
st.markdown("**How much do you know about ML?**")
knowledge_slider = st.slider("1=Beginner to 5=Expert:", 1, 5, 1)
st.write("Your current knowledge level:", knowledge_slider)
st.write("(Keep note of me to reflect after completing the content!)")


# Progress bar to indicate the users learning journey.
st.markdown("# Get started!")
st.markdown("Dive into the sections, experiment with datasets, and watch your skills grow as you progress through this app. Let’s demystify machine learning together and unlock its potential to address real-world challenges!")
st.progress(10)