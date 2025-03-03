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
import os


st.title("Unsupervised Learning Page")

st.markdown("We'll add some examples in later")





# Interactive graphs of the pre/post data for each marker. 

st.subheader("INTRO STUFF ")
st.markdown("Explain unsupervised learning")
      



st.subheader("PCA and K-means Clustering")

st.markdown("PCA simplifies complex datasets by reducing the number of features, while keeping as much of the important information so that the significancy of the data is not affected. First, the data is standardised, so that all features are on the same scale. Then key features are identified, through the combination of original features. Finally, the dimensions are reduced, only the top few features are kept which retains the most significant information.")


st.markdown("PCA is usually followed by a clustering algorithm. K-means clustering is a common way to group data into different categories based on how similar the data points are. It starts with picking the number of groups, with random group centres. The data points are then assigned to the nearest group and the group centres are updates. This is repeated until the best grouping is found.")


st.markdown("**Example:** Explore PCA and clustering on the Toy dataset below.")

toy_dat = pd.read_csv('toy.csv')
toy_dat['encoded_label'] = [1 if i == 'b' else 0 for i in toy_dat['label'].values]  # Makes a = 0, b = 1.
del toy_dat['label']    # Drop labels column as not computer readable.

st.subheader("Raw Toy Dataset:")
st.dataframe(toy_dat.head())
st.markdown("The columns, or features, of the toy dataset represent the variables measured for each data point. Whereas, the data points represent individual samples, with each row in the dataset being a different sample. The plot below displays the first two features of the dataset.")

# Apply PCA
pca = PCA(n_components=2)
toydat_pca = pca.fit_transform(toy_dat)
toy_dat_pca = pd.DataFrame(toydat_pca, columns=["PC1", "PC2"])

<<<<<<< HEAD
# Show original data (first two numerical columns)
st.markdown("**First Two Features**")
fig, ax = plt.subplots()
ax.scatter(toy_dat.iloc[:, 0], toy_dat.iloc[:, 1], alpha=0.5)
ax.set_xlabel(toy_dat.columns[0])
ax.set_ylabel(toy_dat.columns[1])
st.pyplot(fig)

# Show PCA result
st.subheader("After PCA")
st.markdown("PCA transforms the original dataset into a new set of axes, known as principle components. The 1st principle component (PC1) captures the greatest variance in the data, the 2nd principle component (PC2) captures the second greatest variance and so on, capturing less and less variance for each principle component. Below you can see a change in structure and separation of the data points.")
fig, ax = plt.subplots()
ax.scatter(toy_dat_pca["PC1"], toy_dat_pca["PC2"], alpha=0.5, color='red')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

# K-Means clustering
st.subheader("K-Means Clustering")
st.markdown("What number of clusters best fits the transformed toy data?")
num_clusters = st.slider("Select number of clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(toydat_pca)
toy_dat_pca["Cluster"] = kmeans.labels_

# Show clustered data
fig, ax = plt.subplots()
for cluster in range(num_clusters):
    cluster_points = toy_dat_pca[toy_dat_pca["Cluster"] == cluster]
    ax.scatter(cluster_points["PC1"], cluster_points["PC2"], label=f"Cluster {cluster}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
st.pyplot(fig)


st.subheader("INTERACTIVE PLOT TO SEE HOW CLUSTERING WORKS WITH DIFFERENT DATASHAPES")


# normal writing details
multi = '''Data doesnt always come in blobs - it can come in other shapes:    
Different clustering techniques work better than others depending on the shape of the data

The following figure allows the user to produce some data selected randomly from a set of different dataset -  
Lots of them have funky shapes eg data in shape of smiley face (shown below) 

Will allow users to see how clustering works for different data shapes in a *fun* :rainbow[interactive] way 


.


There are three main methods for clustering:
- K-means clustering
- Gaussian Mixture Model (GMM) clustering
- DBSCAN

These different methods often produce different clusters on the same set of data, as we will see below.

K-means produces "blobby" clusters, with the shape of the clusters always in a circular, blobby shape. 
This means if the data has striata, these are often not determined as separate clusters.

K-means involves minimising the distance between the points and each cluster centre. 
GMM involves calculating the probability of each point being in each cluster. 

DBSCAN does not assume the shapes of the clusters, so is often more effective at clustering strangely-shaped data, 
which form obvious clusters to the human eye, but which K-means and GMM may struggle with. 

K-means and GMM both need the number of clusters to be defined by the user. The number of clusters is not always easy to do, 
as we will see below.

'''
st.markdown(multi)

st.image("images/funky_shapes.png", caption="Data with funky shapes", width=600)


#######################################################################################################################################
st.subheader("LOADING IN FUNKY DATA")


def get_data(seed):
    random.seed(seed)
    random_file = random.choice(os.listdir("datasets/cluster data")) 
    # st.write(random_file)

    random_data = pd.read_csv(f"datasets/cluster data/{random_file}")
    random_data.drop(columns=["color"])
    
    # random_noise = np.random.normal(0, 1)
    # random_data += random_noise
    
    # Add noise??
    
    return random_data


if "counter" not in st.session_state:
    initial_seed = random.randint(0, 100)
    st.session_state["counter"] = initial_seed
    st.write(f"Initial seed counter: {initial_seed}")


random_data = get_data(st.session_state["counter"])

# Select and plot random file of funky data
if st.button("Get new data", type="primary"):
    
    st.session_state["counter"] += 1
    random_data = get_data(st.session_state["counter"])

# Option to show data
if st.checkbox('Show dataframe'):
    st.write(random_data.head(5))



noise_std = st.slider("Select noise distribution", 0.0, 20.0, step=1.0 )
rand_noise = np.random.normal(0,noise_std, random_data.shape )

random_data += rand_noise


#######################################################################################################################################
st.subheader("CLUSTER AND PLOT FUNKY DATA")

# have plot next to explanation of limitations of each model. 

col1, col2 = st.columns([0.7, 0.3])


def basic_plot(data, size=10, colour_labels=0, cluster_centres=None):
    """
    Basic scatter plot of some data
    
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(data["x"], data["y"], c=colour_labels, s=size)
    ax.set_aspect("equal")
    
    if isinstance( cluster_centres, np.ndarray):
        ax.scatter(cluster_centres[:, 0], cluster_centres[:, 1], s=size, marker='s', c="cyan")
    
    st.pyplot(fig)



def cluster_plots(counter, technique, X, fig, ax, num_clusters=None):
    """
    Updater function to allow the viewer to slide through the data for each marker.
    Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

    Args:
        marker_num (int): Which marker the viewer wants to see. 
        histogram (str): Which data the viewer wants to see. 
    """
    from sklearn.datasets import make_blobs
    import random
    

    if technique == clustering_technique_options[1]:
        # do K-means
        st.write("running k means")
        
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_centres)
        kmeans.fit(X)
                
        ax.plot(X[:, 0], X[:, 1], '.')
        ax.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 's')

    elif technique == clustering_technique_options[2]:
        # do gmm
        st.write("running gmm")
        
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=num_centres).fit(X)
        labels = gmm.predict(X)
        
        probs = gmm.predict_proba(X)
        size = 15 * probs.max(axis=1) ** 2
        
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=size)

    elif technique == clustering_technique_options[3]:
        # do dbscan
        st.write("dbscan")
        
        from sklearn.cluster import DBSCAN
        
        dbscan = DBSCAN(eps=3, min_samples=5).fit(X)
        labels = dbscan.labels_
        
        st.write(labels)

        fig, ax = plt.subplots(figsize=(4, 4))
        # ax.scatter(*X.T, c=label)
        
        ax.scatter(X[:, 0], X[:, 1], '.', c=labels)
        
    elif technique == clustering_technique_options[4]:
        from sklearn.cluster import HDBSCAN

        hdbscan = HDBSCAN()
        label = hdbscan.fit_predict(X)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(*X.T, c=label)
        ax.axis('equal')
        
        st.write("dbscan")

        ax.plot(X[:, 0], X[:, 1], '.')
    else:
        st.write("No clustering")
        
        ax.plot(X[:, 0], X[:, 1], '.')


    ax.set_xlim(-axis_max, axis_max)
    ax.set_ylim(-axis_max, axis_max)
    st.pyplot(fig, clear_figure=True)
    
    
def kmeans_cluster(data, num_centres):
    kmeans = KMeans(n_clusters=num_centres)
    kmeans.fit(data)
    
    labels = kmeans.labels_
    cluster_centres = kmeans.cluster_centers_
    
    return labels, cluster_centres

def gmm_cluster(data, num_centres):
    st.write("running gmm")
    
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=num_centres).fit(data)
    colour_labels = gmm.predict(data)
    
    probabilities = gmm.predict_proba(data)
    size = 15 * probabilities.max(axis=1) ** 2
    
    return size, colour_labels


def dbscan_cluster(data, eps):
    st.write("running dbscan")
    from sklearn.cluster import DBSCAN
    
    dbscan = DBSCAN(eps=eps, min_samples=5).fit(data)
    labels = dbscan.labels_
    
    num_labels = len(np.unique(labels))
    st.write(f"Number of unique labels: {num_labels}")

    return labels


with col1:

    clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN"]
    technique = st.selectbox('Select a technique for clustering.', clustering_technique_options)
        
    cluster_centres = None
    if technique == clustering_technique_options[1]:
        num_clusters = st.slider("Select number of clusters", 1, 6)
        colour_labels, cluster_centres = kmeans_cluster(random_data, num_clusters)
        size = 15
        
    elif technique == clustering_technique_options[2]:
        num_clusters = st.slider("Select number of clusters", 1, 6)
        size, colour_labels = gmm_cluster(random_data, num_clusters)
        
    elif technique == clustering_technique_options[3]:
        st.write("need to be dbscan")
        eps = st.slider("Eps", 1, 15)
        colour_labels = dbscan_cluster(random_data, eps)
        size=15
        
    else:
        size = 15
        colour_labels = np.zeros(len(random_data))
        
    basic_plot(random_data, size, colour_labels, cluster_centres)
        
        
    
# st.write("previous data slider bool:", st.session_state["previous_data_slider"])
    st.write("counter:", st.session_state["counter"])
   

with col2:
    st.write("lil explanation of this dataset")   
   
 
# # Option to look at past data - do later
# # Need to make permanent
# if "previous_data_slider" not in st.session_state:
#     st.session_state["previous_data_slider"] = False
    
# if st.button("Review previous data", type = "secondary"):
#     st.session_state["previous_data_slider"] = True
 
 
 
# if st.session_state["previous_data_slider"] == True:
#     if st.session_state["counter"] == 0:
#         st.write("No previous data to re-explore...")
#         previous_seed = st.session_state["counter"]
#     else:
#         max_counter = st.session_state["counter"]
#         previous_seed = st.slider("Previous seeds", 0, max_counter)
    
#     # do func
#     old_data = get_old_data()
    
#     # update to new func 
#     cluster_plots(previous_seed, technique, old_data, fig, ax, num_clusters)

# else:
#     # NEEDS DATA
#     data = random_data
    
#     # update to new func
#     cluster_plots(st.session_state["counter"], technique, data, fig, ax, num_clusters)













# def update_marker3(counter, technique2, X, fig, ax):
#     """
#     Updater function to allow the viewer to slide through the data for each marker.
#     Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

#     Args:
#         marker_num (int): Which marker the viewer wants to see. 
#         histogram (str): Which data the viewer wants to see. 
#     """
#     from sklearn.datasets import make_blobs
#     import random
    

#     # ax.plot(X[:, 0], X[:, 1], '.')
    
    
    
#     if technique2 == clustering_technique_options[1]:
#         # do K-means
#         st.write("running k means")
        
#         from sklearn.cluster import KMeans

#         kmeans = KMeans(n_clusters=num_centres)
#         kmeans.fit(X)
        
#         # x_vals = X[:, 0]
#         # y_vals = X[:, 1]
#         # x_centre_vals = kmeans.cluster_centers_[:, 0]
#         # y_centre_vals = kmeans.cluster_centers_[:, 1]
        
#         ax.plot(X[:, 0], X[:, 1], '.')
#         ax.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 's')

#     elif technique2 == clustering_technique_options[2]:
#         # do gmm
#         st.write("running gmm")
        
#         from sklearn.mixture import GaussianMixture

#         gmm = GaussianMixture(n_components=num_centres).fit(X)
#         labels = gmm.predict(X)
        
#         probs = gmm.predict_proba(X)
#         size = 10 * probs.max(axis=1) ** 2
        
#         ax.scatter(X[:, 0], X[:, 1], c=labels, s=size)

#     elif technique2 == clustering_technique_options[3]:
#         # do dbscan
#         st.write("dbscan")

#         ax.plot(X[:, 0], X[:, 1], '.')
        
#     else:
#         st.write("None")
        
#         ax.plot(X[:, 0], X[:, 1], '.')


#     ax.set_xlim(-axis_max, axis_max)
#     ax.set_ylim(-axis_max, axis_max)
#     st.pyplot(fig, clear_figure=True)
    
    
    
# def make_my_blobs(seed):
#     random.seed(seed)
#     num_centres = randint(3, 6)
#     centre_max_val = 4
#     std = 1
#     axis_max = centre_max_val + 2*std
#     X, y = make_blobs(n_samples=300, n_features=2, centers=num_centres, center_box=(-centre_max_val, centre_max_val), cluster_std=std, random_state=seed)
#     return X, axis_max


# st.subheader("FIGURE")


# if "counter" not in st.session_state:
#     st.write("counter not in...............")
#     st.session_state["counter"] = 0


# fig, ax = plt.subplots(figsize=(6, 4))

# # Unfinished buttons which will allow selecting of different data
# left, middle, right = st.columns(3)
# if left.button("Get blobby  data", use_container_width=True):
#     left.markdown("Producing some blobby data.")
# if middle.button("Get moony data", use_container_width=True):
#     middle.markdown("Producing some moony data.")
# if right.button("Get moony/blobby data", use_container_width=True):
#     right.markdown("Producing some moony/blobby data.")

# if st.button("Get new data", type="primary"):
#     st.session_state["counter"] += 1
#     # new_X = make_my_blobs(st.session_state["counter"])[0]
#     # update_marker3(st.session_state["counter"], new_X, fig, ax)

# # Option to look at past data
# # Need to make permanent
# if "previous_data_slider" not in st.session_state:
#     st.session_state["previous_data_slider"] = False
    
# if st.button("Review previous data", type = "secondary"):
#     st.session_state["previous_data_slider"] = True
   
 
# clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN"]
# technique2 = st.selectbox('Select a technique for clustering.', clustering_technique_options)
    
# st.write("previous data slider bool:", st.session_state["previous_data_slider"])
# st.write("counter:", st.session_state["counter"])
    
    
 
# if st.session_state["previous_data_slider"] == True:
#     if st.session_state["counter"] == 0:
#         st.write("No previous data to re-explore...")
#     else:
#         max_counter = st.session_state["counter"]
#         previous_seed = st.slider("Previous seeds", 0, max_counter)
        
#     X, axis_max = make_my_blobs(previous_seed)
#     update_marker3(previous_seed, technique2, X, fig, ax)

# else:
#     X, axis_max = make_my_blobs(st.session_state["counter"])
#     update_marker3(st.session_state["counter"], technique2, X, fig, ax)




 




st.subheader("CLUSTER SELECTION")


st.write("selecting the correct number of clusters can be done in some way")
st.write("brief overview")
st.write("however we wont go into much detail here")
st.write("here are some links tho")






