import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st 
import random
from sklearn.cluster import KMeans
import time
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, HDBSCAN

def plot_clusters(pc, fig=None, ax=None, num_clusters = 2):
      
    if fig == None:
        fig,ax = plt.subplots()
      
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pc[['PC1', 'PC2']])
    pc["Cluster"] = kmeans.labels_
    
    for cluster in range(2):
        subset = pc[pc["Cluster"] == cluster]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7, s=30)

    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=100, c="black", label="Cluster Centres")

    ax.set(xlabel="PC1", ylabel="PC2", title=f"K-Means Clustering (k={num_clusters})")
    ax.legend()
    st.pyplot(fig)


def run_kmeans_animation(X, n_clusters=2):
    button_placeholder = st.empty()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize cluster centers, using random seed
    np.random.seed(0)
    i = np.random.randint(0, X.shape[0], size=n_clusters)
    centres = X[i]

    old_centres = []

    max_iterations = 10
    stop = False

    # col1, col2 = st.columns([0.99, 0.01])
    placeholder = st.empty()
    
    # Plot initial data, with two (random) points chosen
    ax.scatter(X[:, 0], X[:, 1], marker='.')
    ax.scatter(centres[:, 0], centres[:, 1], marker='*', s=100, color='black', label='Current Centres', zorder=3 )
    ax.legend()
    placeholder.pyplot(fig)

    # run when button pressed
    if button_placeholder.button("Run animation", type="primary"):
        
        # loop through number of iterations
        for iteration in range(max_iterations):
            if not stop:
                ax.clear()
                
                # update centres
                new_centres = centres
                old_centres.append(centres)
                
                # plot previous centres
                for i, centres in enumerate(old_centres):
                    # fade previous centres more if they are older
                    alpha = 0.5 * (1+ ((i+1)/len(old_centres) ))
                    
                    ax.scatter(centres[:, 0], centres[:, 1], marker='x', color='black', alpha=alpha, s=15, zorder=2)
                
                # Calculate distance from each point to each centre
                vectors = X[:, np.newaxis] - centres
                
                # calculate magnitudes of vectors
                magnitudes = np.linalg.norm(vectors, axis=-1)
                
                labels = magnitudes.argmin(axis=1)
                
                # ax.scatter(X[:, 0], X[:, 1], marker='.', c=labels)
                for cluster in range(2):
                    subset = X[labels == cluster]
                    ax.plot(subset[:, 0], subset[:, 1], alpha=1, marker='.', linestyle='', zorder=1)
                
                # calculate average of each group (plot)
                new_centres = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
                
                # stop if centres are close
                stop = np.allclose(centres, new_centres, atol=0.1)
                
                # set new averages as new centre points (plot)
                centres = new_centres
                
                ax.scatter(centres[:, 0], centres[:, 1], marker='*', s=100, color='black', label='Current Centres', zorder=3 )
                ax.legend()
                
                placeholder.pyplot(fig)
                time.sleep(1.5)  
            else:
                st.write(f'Clustering Complete! Final Iteration: {iteration}')
                break
            


def get_data(seed, cluster_dict): 
    """
    Get a random file from the funky-shaped datasets. 

    Args:
        seed (int): random seed
        cluster_dict (dict): dictionary of cluster filenames

    Returns:
        random_file: name of file
        random_data: data of file
    """
    random.seed(seed)
    random_file = random.choice(list(cluster_dict.keys())) 

    random_data = pd.read_csv(f"datasets/cluster data/{random_file}")
    
    # drop color column if it exists
    random_data = random_data.drop(columns=["color"], errors='ignore')
    
    # normalise data    
    random_data = pd.DataFrame( StandardScaler().fit_transform(random_data.values), columns=random_data.columns )
            
    return random_file, random_data



def kmeans_cluster(data, num_centres):
    """
    Cluster data using kmeans clustering

    Args:
        data (df): input data to cluster
        num_centres (int): Number of clusters

    Returns:
        labels: labels for each datapoint of which cluster they are in
        cluster_centres: coordinates of cluster centres
    """
    kmeans = KMeans(n_clusters=num_centres)
    kmeans.fit(data)
    
    labels = kmeans.labels_
    cluster_centres = kmeans.cluster_centers_
    
    return labels, cluster_centres



def gmm_cluster(data, num_centres):     
    """
    Cluster data using gmm clustering

    Args:
        data (df): input data to cluster
        num_centres (int): Number of clusters

    Returns:
        size: Size proportional to probabilty of data being in given cluster
        colour_labels: labels for each datapoint of which cluster they are in
    """   

    gmm = GaussianMixture(n_components=num_centres).fit(data)
    colour_labels = gmm.predict(data)
    
    probabilities = gmm.predict_proba(data)
    size = 15 * probabilities.max(axis=1) ** 2
    
    return size, colour_labels


def dbscan_cluster(data, eps, min_samples):
    """
    Cluster data using dbscan clustering

    Args:
        data (df): input data to cluster
        eps (int): Max distance for points to be neighbours.
        min_samples (int): Minimum number of points needed to form a cluster.

    Returns:
        labels: labels for each datapoint of which cluster they are in
    """   
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = dbscan.labels_
    
    num_labels = len(np.unique(labels))
    st.write(f"Number of unique labels: {num_labels}")

    return labels


def hdbscan_cluster(data):
    """
    Cluster data using hdbscan clustering

    Args:
        data (df): input data to cluster

    Returns:
        labels: labels for each datapoint of which cluster they are in
    """   
    
    labels = HDBSCAN().fit_predict(data)

    return labels


def basic_plot(data, ax, size=10, colour_labels=None, cluster_centres=None):
    """
    Plot of data with coloured labels representing different clusters

    Args:
        data (df): input data to plot
        ax (ax): axis to plot on
        size (int, optional): Size of scatter points. Defaults to 10.
        colour_labels (arr, optional): labels of colours describing clusters. Defaults to None.
        cluster_centres (arr, optional): coordinates of cluster centres. Defaults to None.
    """
    if colour_labels is not None:
        colour_labels=colour_labels
    else:
        colour_labels=np.zeros(len(data))
    
    ax.clear()
    
    ax.scatter(data["x"], data["y"], c=colour_labels, s=size, cmap="gist_rainbow")
    ax.set_aspect("equal")
    
    # Plot cluster centres if they exist
    if isinstance( cluster_centres, np.ndarray):
        ax.scatter(cluster_centres[:, 0], cluster_centres[:, 1], s=size, marker='s', c="black")


def pros_and_cons(multi_pros, multi_cons):
    """
    Write pros and cons of a method in two columns

    Args:
        multi_pros (str): Multi line string containing list of pros
        multi_cons (str): multi line string containing list of cons
    """
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Pros")
    with col2:
        st.header("Cons")
    with st.container(border=True, height=200):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(multi_pros)
        with col2:
            st.write(multi_cons)
