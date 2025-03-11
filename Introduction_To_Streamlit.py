# this is what to run on the commandline
# streamlit run .\Introduction_To_Streamlit.py



import streamlit as st
import numpy as np
import pandas as pd


# st.markdown("# Welcome page 🎈")
# st.sidebar.markdown("# Main page 🎈")


# have a lil quiz for "are you ready to get started!"

st.title("Machine Learning Teaching Material")
st.write("*Give a brief introduction about the project, contents of whats to come, and how to interact with the different kinds of widgets.  ")


st.write("Hi! Welcome to our project!")
st.write("Throughout the rest of this project, you will be taken on a journey through machine learning and data science.")
st.write("Each page (shown along the left hand side <--) covers a different part of machine learning")

st.write("The first page covers some explanations between ML, AI, and other imprtant concepts. If you've never covered these topics before,\
         make sure to have a look through this page so you're comfortable with the different parts.")
intro_multi = """
    If you're already an expert in the definitions feel free to skip on past this page, onto the more in-depth explanations and examples!
    But make sure you do understand these basics because they're really important!!
"""

st.write(intro_multi)


widgets_multi = """
    Throughout the rest of the notebook, there are many examples.  
    These examples are interactive, to make the learning more fun, effective and ENGAGING!
    
    Below we have a few very simple demonstrations of some of the widgets,  
    to ensure you know how to operate them throughout the rest of this project.
    
"""

x = np.arange(1,100,10)
y=2*x
# data1 = [x,y]
data1 = pd.DataFrame({
  'first column': x,
  'second column': y
})

# print(data1)

st.line_chart(data1, x="first column", y="second column")


# widget
slider = st.slider('x')  
st.write(slider, 'squared is', slider * slider)


# checkbox
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data
    
    
# dropdown menu
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

# add sidebar
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)