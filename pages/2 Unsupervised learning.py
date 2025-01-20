import streamlit as st 
import altair as alt

st.title("Unsupervised Learning Page")



st.markdown("We'll add some examples in later")



# The imports used throughout this notebook
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.preprocessing import scale


# Interactive graphs of the pre/post data for each marker. 


st.markdown("Explain unsupervised learning")
      

st.markdown("IDEA: \n potentially have a plot which produces random data into a random number of clusters. The student then has to flick, through, changing the number of clusters until the graph looks right. Makes it a more unique experience for theh student as they have their own set of data. Allow them to reset and produce another random set of data so they can do it as often as they like. ")


st.markdown(" \" Here's an example which allows exploring of unsupervised learning techniques PCA and clustering \" ")


# Load in the data provided by TheraTech.
post_data = pd.read_csv('pages/post_cancer_markers.csv')
pre_data = pd.read_csv('pages/pre_cancer_markers.csv')

# List of just the column names. Used later. 
marker_names = post_data.columns[:6]

# A sample of one of the dataframes to show how the data is provided. 
# Only the first 5 rows of 2000 are shown.
st.write(post_data.head())



# 
st.markdown("animation showing how k means clustering works")

st.markdown("Vary number of clusters (maybe use different data than the cancer marker data)")





# Interactive plotting using above updater function.

# List of the different options for data to view.
viewing_options = ['Pre and Post data', 'Pre data', 'Post data']

# Allows viewing of full slider name.
# style = {'description_width': 'initial'} 
 
# # Initialises figure 


def update_marker(marker_num):
    """
    Updater function to allow the viewer to slide through the data for each marker.
    Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

    Args:
        marker_num (int): Which marker the viewer wants to see. 
        histogram (str): Which data the viewer wants to see. 
    """
    
    # Creates arrays of the pre and post data for the required marker
    marker_i_pre = pre_data[f'marker_{marker_num}']
    marker_i_post = post_data[f'marker_{marker_num}']
    pre_min = np.min(marker_i_pre)
    pre_max = np.max(marker_i_pre)
    post_min = np.min(marker_i_post)
    post_max = np.max(marker_i_post)

    df = pd.DataFrame({"pre column": marker_i_pre, "post column": marker_i_post})
    # st.scatter_chart(df, x="pre column")
    
    c = (
        alt.Chart(df)
        .mark_circle()
        .encode(
        alt.X('pre column', scale=alt.Scale(domain=[0.99*pre_min, 1.01*pre_max], nice=False)),
        alt.Y('post column', scale=alt.Scale(domain=[0.99*post_min, 1.01*post_max], nice=False)),
        )
    )

    st.altair_chart(c, use_container_width=False)
    

slider = st.slider(label='Marker number:', min_value=0, max_value=5, step=1) # , style=style)  

st.write(slider)

# Makes widget, with an integer slider to change marker number, and a drop down menu to see the different data distributions.  
update_marker(slider)





st.markdown("Comparison and correlation between different markers?")

st.markdown("Show output of PCA using 2/3 components")

st.markdown("Bar codes of variation coverered using differnet number of components")



