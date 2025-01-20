# this is what to run on the commandline
# streamlit run .\streamlit_test.py



import streamlit as st
import numpy as np
import pandas as pd


# st.markdown("# Welcome page 🎈")
# st.sidebar.markdown("# Main page 🎈")


st.title("Machine Learning Teaching Material")
# st.write("Hopefully this works")

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