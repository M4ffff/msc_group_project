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