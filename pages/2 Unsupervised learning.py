import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.preprocessing import scale
import streamlit as st 
import altair as alt
from sklearn.datasets import make_blobs
from random import randint
import random
from sklearn.cluster import KMeans

st.title("Unsupervised Learning Page")



st.markdown("We'll add some examples in later")





# Interactive graphs of the pre/post data for each marker. 

st.subheader("INTRO STUFF ")
st.markdown("Explain unsupervised learning")
      

st.markdown("IDEA: \n potentially have a plot which produces random data into a random number of clusters. The student then has to flick, through, changing the number of clusters until the graph looks right. Makes it a more unique experience for theh student as they have their own set of data. Allow them to reset and produce another random set of data so they can do it as often as they like. ")


st.markdown(" \" Here's an example which allows exploring of unsupervised learning techniques PCA and clustering \" ")


st.subheader("SCATTER TEST ")

from sklearn.datasets import make_blobs
from random import randint

num_centres = randint(3, 6)
centre_max_val = 4
std = 1
axis_max = centre_max_val + 2*std
X, y = make_blobs(n_samples=300, n_features=2, centers=num_centres, center_box=(-centre_max_val, centre_max_val), cluster_std=std, random_state=2)

chart_data = pd.DataFrame(X, columns=["a", "b"])

c = (
   alt.Chart(chart_data)
   .mark_circle()
   .encode(x=alt.X("a", scale=alt.Scale(domain=[-axis_max, axis_max])),
            y=alt.Y("b", scale=alt.Scale(domain=[-axis_max, axis_max])) 
    )
)

st.altair_chart(c, use_container_width=True)


st.subheader("PYPLOT TEST ")


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1, random_state=0)
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
size = 10 * probs.max(axis=1) ** 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X[:, 0], X[:, 1], c=labels, s=size)


st.pyplot(fig)



st.subheader("MAP TEST ")




# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)










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


















# Interactive plotting using above updater function.

# List of the different options for data to view.
viewing_options = ['Pre and Post data', 'Pre data', 'Post data']

# Allows viewing of full slider name.
# style = {'description_width': 'initial'} 
 
# # Initialises figure 


# def update_marker2():
#     """
#     Updater function to allow the viewer to slide through the data for each marker.
#     Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

#     Args:
#         marker_num (int): Which marker the viewer wants to see. 
#         histogram (str): Which data the viewer wants to see. 
#     """
#     from sklearn.datasets import make_blobs
    
    
#     num_centres = 3
#     centre_max_val = 4
#     std = 1
#     axis_max = centre_max_val + 2*std
#     X, y = make_blobs(n_samples=300, n_features=2, centers=num_centres, center_box=(-centre_max_val, centre_max_val), cluster_std=std, random_state=3)

#     chart_data = pd.DataFrame(X, columns=["a", "b"])
    
    
#     clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN"]
#     technique = st.selectbox('Select a technique for clustering.', clustering_technique_options)
    
#     if technique == clustering_technique_options[1]:
#         # do K-means
#         st.write("running k means")
        
#         from sklearn.cluster import KMeans

#         kmeans = KMeans(n_clusters=num_centres)
#         kmeans.fit(X)
        
#         # x_vals = X[:, 0]
#         # y_vals = X[:, 1]
#         # x_centre_vals = kmeans.cluster_centers_[:, 0]
#         # y_centre_vals = kmeans.cluster_centers_[:, 1]
        
#         points = (
#         alt.Chart(chart_data)
#         .mark_circle()
#         .encode(x=alt.X("a", scale=alt.Scale(domain=[-axis_max, axis_max])),
#                     y=alt.Y("b", scale=alt.Scale(domain=[-axis_max, axis_max])) 
#             )
#         )
        
#         centres_data = pd.DataFrame(kmeans.cluster_centers_, columns = ["a", "b"])
        
#         centres = (
#             alt.Chart(centres_data)
#             .mark_point(size=100, color="red", shape="x") 
#             .encode(
#                 x="a",
#                 y="b"
#             )
#         )
        
#         c = points+centres


#     elif technique == clustering_technique_options[2]:
#         # do gmm
#         st.write("running gmm")
        
#         from sklearn.mixture import GaussianMixture

#         gmm = GaussianMixture(n_components=4).fit(X)
#         labels = gmm.predict(X)
        
        
#         probs = gmm.predict_proba(X)
#         size = 10 * probs.max(axis=1) ** 2
        
        
#         chart_data["labels"] = labels
#         chart_data["size"] = size
        
#         c = (
#         alt.Chart(chart_data)
#         .mark_circle()
#         .encode(x=alt.X("a", scale=alt.Scale(domain=[-axis_max, axis_max])),
#                     y=alt.Y("b", scale=alt.Scale(domain=[-axis_max, axis_max])), 
#                     color = "labels",
#                     size = "size"
#             )
#         )

#     elif technique == clustering_technique_options[3]:
#         # do dbscan
#         st.write("dbscan")

#         c = (
#         alt.Chart(chart_data)
#         .mark_circle()
#         .encode(x=alt.X("a", scale=alt.Scale(domain=[-axis_max, axis_max])),
#                     y=alt.Y("b", scale=alt.Scale(domain=[-axis_max, axis_max])) 
#             )
#         )
        
#     else:
#         st.write("None")
#         c = (
#         alt.Chart(chart_data)
#         .mark_circle()
#         .encode(x=alt.X("a", scale=alt.Scale(domain=[-axis_max, axis_max])),
#                     y=alt.Y("b", scale=alt.Scale(domain=[-axis_max, axis_max])) 
#             )
#         )

#     st.altair_chart(c, use_container_width=True)
    
    
    


# # slider = st.slider(label='Marker number:', min_value=0, max_value=5, step=1) # , style=style)  

# # st.write(slider)

# # Makes widget, with an integer slider to change marker number, and a drop down menu to see the different data distributions.  

# print("just before calling function")
# update_marker2()






def update_marker3(counter, technique2, X, fig, ax):
    """
    Updater function to allow the viewer to slide through the data for each marker.
    Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

    Args:
        marker_num (int): Which marker the viewer wants to see. 
        histogram (str): Which data the viewer wants to see. 
    """
    from sklearn.datasets import make_blobs
    import random
    

    # ax.plot(X[:, 0], X[:, 1], '.')
    
    
    
    if technique2 == clustering_technique_options[1]:
        # do K-means
        st.write("running k means")
        
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_centres)
        kmeans.fit(X)
        
        # x_vals = X[:, 0]
        # y_vals = X[:, 1]
        # x_centre_vals = kmeans.cluster_centers_[:, 0]
        # y_centre_vals = kmeans.cluster_centers_[:, 1]
        
        ax.plot(X[:, 0], X[:, 1], '.')
        ax.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 's')

    elif technique2 == clustering_technique_options[2]:
        # do gmm
        st.write("running gmm")
        
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=num_centres).fit(X)
        labels = gmm.predict(X)
        
        probs = gmm.predict_proba(X)
        size = 10 * probs.max(axis=1) ** 2
        
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=size)

    elif technique2 == clustering_technique_options[3]:
        # do dbscan
        st.write("dbscan")

        ax.plot(X[:, 0], X[:, 1], '.')
        
    else:
        st.write("None")
        
        ax.plot(X[:, 0], X[:, 1], '.')


    ax.set_xlim(-axis_max, axis_max)
    ax.set_ylim(-axis_max, axis_max)
    st.pyplot(fig, clear_figure=True)
    
    
    
def make_my_blobs(seed):
    random.seed(seed)
    num_centres = randint(3, 6)
    centre_max_val = 4
    std = 1
    axis_max = centre_max_val + 2*std
    X, y = make_blobs(n_samples=300, n_features=2, centers=num_centres, center_box=(-centre_max_val, centre_max_val), cluster_std=std, random_state=seed)
    return X, axis_max


if "counter" not in st.session_state:
    st.write("counter not in...............")
    st.session_state["counter"] = 0


fig, ax = plt.subplots(figsize=(6, 4))

# Unfinished buttons which will allow selecting of different data
left, middle, right = st.columns(3)
if left.button("Get blobby  data", use_container_width=True):
    left.markdown("Producing some blobby data.")
if middle.button("Get moony data", icon="😃", use_container_width=True):
    middle.markdown("Producing some moony data.")
if right.button("Get moony/blobby data", icon=":material/mood:", use_container_width=True):
    right.markdown("Producing some moony/blobby data.")

if st.button("Get new data", type="primary"):
    st.session_state["counter"] += 1
    # new_X = make_my_blobs(st.session_state["counter"])[0]
    # update_marker3(st.session_state["counter"], new_X, fig, ax)

# Option to look at past data
# Need to make permanent
if "previous_data_slider" not in st.session_state:
    st.session_state["previous_data_slider"] = False
    
if st.button("Review previous data", type = "secondary"):
    st.session_state["previous_data_slider"] = True
   
 
clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN"]
technique2 = st.selectbox('Select a technique for clustering.', clustering_technique_options)
    
st.write("previous data slider bool:", st.session_state["previous_data_slider"])
st.write("counter:", st.session_state["counter"])
    
    
 
if st.session_state["previous_data_slider"] == True:
    if st.session_state["counter"] == 0:
        st.write("No previous data to re-explore...")
    else:
        max_counter = st.session_state["counter"]
        previous_seed = st.slider("Previous seeds", 0, max_counter)
        
    X, axis_max = make_my_blobs(previous_seed)
    update_marker3(previous_seed, technique2, X, fig, ax)

else:
    X, axis_max = make_my_blobs(st.session_state["counter"])
    update_marker3(st.session_state["counter"], technique2, X, fig, ax)




 













