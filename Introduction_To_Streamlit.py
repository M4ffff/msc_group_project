# this is what to run on the commandline
# streamlit run .\Introduction_To_Streamlit.py



import streamlit as st
import numpy as np
import pandas as pd


# st.markdown("# Welcome page 🎈")
# st.sidebar.markdown("# Main page 🎈")


# have a lil quiz for "are you ready to get started!"

st.title("Machine Learning Interactive Platform")
# st.write("*Give a brief introduction about the project, contents of whats to come, and how to interact with the different kinds of widgets.  ")

st.subheader("Hellooooo!")
st.write("Welcome to our Streamlit :streamlit: project! ")

st.write("Throughout the rest of this project, you will be taken on a :rainbow[***journey***] through **machine learning** :robot_face: and **data science**. :male-scientist:")
st.write("Each page (shown along the **left hand side** :point_left:) covers a different part of machine learning.")

st.write("The first page covers some explanations of different types of machine learning, artificial intelligence, and other important concepts. If you've never covered these topics before,\
         make sure to have a look through this page so these concepts aren't alien to you throughout the rest of this resource. :alien:")
intro_multi = """
    If you're already an expert in the definitions feel free to skip on past this page, onto the more **in-depth** :nerd_face: explanations and examples!
    But make sure you do understand these basics because they're really important :heavy_exclamation_mark: :heavy_exclamation_mark:
"""

st.write(intro_multi)


widgets_multi = """
    Throughout the rest of the notebook, there are many examples.\
    These examples are interactive, to make the learning more :blue[fun], :violet[effective] and :rainbow[ENGAGING]!
    
    Below we have a few very simple demonstrations of some of the widgets,\
    to ensure you know how to operate them throughout the rest of this project. :male-scientist:
"""

st.subheader("Widget demonstration")
st.write(widgets_multi)

x = np.arange(1,100,10)
y=2*x

st.subheader("Slider")
# widget
slider_val = st.slider('x', 1, 60)  
    
snake = '<' + slider_val*'=' + '8<'
st.write(snake)
if slider_val == 1:
    st.write("What is that?")
else:
    hs = slider_val*'h'
    st.write(f"a{hs} a snake")


st.subheader("Checkbox")
# checkbox
if st.checkbox('Show dataframe'):
    st.image("images/_intro_page/dataframe.png")
    st.write("Sorry, I meant data *in* a frame here!")
    

st.subheader("Quiz")
my_number = 4
question_one = st.radio(
    "What number am I thinking of?", (np.arange(1,7)), index=None)
    
if question_one == 4:
    st.success("Yeah! Great number :100:")
elif question_one == 1:
    st.error("SO WRONG :x:")
elif question_one == 2:
    st.error("You're halfway there!")
elif question_one == 3 or question_one == 5:
    st.error("So close yet so far :disappointed:")
elif question_one == 6:
    st.error("Ew, no.")
else:
    st.write("")
    
    
    
# dropdown menu
animals = ['dog', 'giraffe', 'tiger', 'walrus']

st.subheader("Select box")
option = st.selectbox(
    'Pick an animal?',
     animals, index=None)

if option in animals:
    st.image(f"images/_intro_page/{option}.png")


# expander
st.subheader("Expander")
with st.expander("Expand me!"):
    st.image("images/_intro_page/ghost.png")

