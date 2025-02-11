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
      



st.subheader("PCA and K-means Clustering")

st.markdown("PCA simplifies complex datasets by reducing the number of features, while keeping as much of the important information so that the significancy of the data is not affected. First, the data is standardised, so that all features are on the same scale. Then key features are identified, through the combination of original features. Finally, the dimensions are reduced, only the top few features are kept which retains the most significant information.")


st.markdown("PCA is usually followed by a clustering algorithm. K-means clustering is a common way to group data into different categories based on how similar the data points are. It starts with picking the number of groups, with random group centres. The data points are then assigned to the nearest group and the group centres are updates. This is repeated until the best grouping is found.")





st.markdown("**Example:** Explore PCA and clustering on the Toy dataset below.")
st.markdown("**Raw Toy data:**")
from sklearn.preprocessing import StandardScaler

toy_dat = pd.read_csv('toy.csv')
toy_dat['encoded_label'] = [1 if i == 'b' else 0 for i in toy_dat['label'].values]  ## Makes a = 0, b = 1.
del toy_dat['label']    ## Drop labels column as not computer readable.

axis_max = max(toy_dat["x"].max(), toy_dat["y"].max()) + 2

c = (
    alt.Chart(toy_dat)
    .mark_circle()
    .encode(x=alt.X("x", scale=alt.Scale(domain=[-axis_max, axis_max])),
            y=alt.Y("y", scale=alt.Scale(domain=[-axis_max, axis_max])),
           )
)

st.altair_chart(c, use_container_width=True)

st.markdown("**Applying PCA:**")
cols = st.multiselect("**Select numerical columns for PCA:**", toy_dat.columns)

if len(cols) >= 2:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(toy_dat[cols])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    
    pca_df["Index"] = toy_dat.index
    
    pca_chart = (
        alt.Chart(pca_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("PC1", scale=alt.Scale(zero=False)),
            y=alt.Y("PC2", scale=alt.Scale(zero=False)),
            tooltip=["Index", "PC1", "PC2"]
        )
    )
    st.altair_chart(pca_chart, use_container_width=True)
    
    explained_variance = pca.explained_variance_ratio_ * 100
    st.write(f"Explained Variance: **PC1 = {explained_variance[0]:.2f}%**, **PC2 = {explained_variance[1]:.2f}%**")
 

st.markdown("**K-means Clustering on Principle Components:**")

k = st.slider("Select a number of clusters:", min_value=2, max_value=10, value=3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
pca_df["Cluster"] = kmeans.fit_predict(pca_df)

axis_max = max(pca_df["PC1"].max(), pca_df["PC2"].max()) + 2

cluster_plot = (
    alt.Chart(pca_df)
    .mark_circle()
    .encode(x=alt.X("PC1", scale=alt.Scale(domain=[-axis_max, axis_max])),
            y=alt.Y("PC2", scale=alt.Scale(domain=[-axis_max, axis_max])),
            color="Cluster:N",
            tooltip=("PC1", "PC2", "Cluster")
           )
)

st.altair_chart(cluster_plot, use_container_width=True)




st.subheader("INTERACTIVE PLOT TO SEE HOW CLUSTERING WORKS WITH DIFFERENT DATASHAPES")


# normal writing details
multi = '''Data doesnt always come in blobs - it can come in other shapes:    
Different clustering techniques work better than others depending on the shape of the data

The following figure allows the user to produce some data selected randomly from a set of different dataset -  
Lots of them have funky shapes eg data in shape of smiley face (shown below) 

Will allow users to see how clustering works for different data shapes in a *fun* :rainbow[interactive] way 
'''
st.markdown(multi)

st.image("images/funky_shapes.png", caption="Data with funky shapes", width=600)


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


st.subheader("FIGURE")


if "counter" not in st.session_state:
    st.write("counter not in...............")
    st.session_state["counter"] = 0


fig, ax = plt.subplots(figsize=(6, 4))

# Unfinished buttons which will allow selecting of different data
left, middle, right = st.columns(3)
if left.button("Get blobby  data", use_container_width=True):
    left.markdown("Producing some blobby data.")
if middle.button("Get moony data", use_container_width=True):
    middle.markdown("Producing some moony data.")
if right.button("Get moony/blobby data", use_container_width=True):
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




 













