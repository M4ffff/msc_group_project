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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Unsupervised Learning Page")


# Interactive graphs of the pre/post data for each marker. 

st.markdown("This page explores unsupervised learning techniques. Unsupervised learning is a type of machine learning where the algorithm learns patterns from unlabelled data, uncovering hidden structures without predefined categories. This approach is particularly useful for exploring large datasets and discovering relationships or groupings within the data. On this page we firstly focus on Principle Component Analysis (PCA) and K-means clustering. By using example datasets, we can demonstrate how PCA can reduce dimensionality for easier visualisation, and how K-means helps identify clusters in data. We also explore how clustering can be applied to data with more complex shapes, highlighting the versatility of these techniques.")
      



st.subheader("PCA and K-means Clustering")

st.markdown("PCA simplifies complex datasets by reducing the number of features, while keeping as much of the important information so that the significancy of the data is not affected. First, the data is standardised, so that all features are on the same scale. Then key features are identified, through the combination of original features. Finally, the dimensions are reduced, only the top few features are kept which retains the most significant information.")

st.markdown("PCA is usually followed by a clustering algorithm. K-means clustering is a common way to group data into different categories based on how similar the data points are. It starts with picking the number of groups, with random group centres. The data points are then assigned to the nearest group and the group centres are updates. This is repeated until the best grouping is found.")

st.markdown("**Example:** Explore PCA and K-means clustering on the breast cancer dataset below.")

bc_dat = pd.read_csv('pages/breast-cancer.csv')
scaled_bc = StandardScaler().fit_transform(bc_dat[bc_dat.columns[1:]])


st.subheader("Raw Breast Cancer Dataset")
bc_dat = pd.read_csv('breast-cancer.csv')
scaled_bc = StandardScaler().fit_transform(bc_dat[bc_dat.columns[1:]])
st.dataframe(bc_dat.head())
st.markdown("The columns, or features, of the breast cancer dataset represent the variables measured for each data point. Whereas, the data points represent individual samples, with each row in the dataset being a different sample.")

st.markdown("Have a look at how the different features of the dataset interact with eachother below!")


# Show original data (numerical columns)
numeric_columns = bc_dat.select_dtypes(include=['number']).columns.tolist()
x_axis = st.selectbox("Select x-axis:", numeric_columns)
y_axis = st.selectbox("Selection y-axis:", numeric_columns)

fig, ax = plt.subplots()
sns.scatterplot(data=bc_dat, x=x_axis, y=y_axis, hue=bc_dat.columns[0], ax=ax, alpha=0.7)
ax.set(xlabel=x_axis, ylabel=y_axis, title=f"{x_axis} vs {y_axis}")
st.pyplot(fig)


# Apply PCA
st.subheader("After PCA")
pca = PCA()
transformed_bc = pca.fit_transform(scaled_bc)
pc = pd.DataFrame(transformed_bc, columns=['PC{}'.format(i + 1) for i in range(transformed_bc.shape[1])])
pc['Diagnosis'] = bc_dat['Diagnosis']
st.dataframe(pc[['Diagnosis', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']].head())

# Check variance ratio
pc_var = pca.fit(scaled_bc)
fig, ax = plt.subplots()
ax.plot(np.cumsum(pc_var.explained_variance_ratio_))
ax.set(xlabel="Number of Components", ylabel="Cumulative Explained Variance")
st.pyplot(fig)

st.markdown("PCA transforms the original dataset into a new set of axes, known as principle components. The 1st principle component (PC1) captures the greatest variance in the data, the 2nd principle component (PC2) captures the second greatest variance and so on, capturing less and less variance for each principle component. For this example we'll focus on the first two principle components, as that is where the majority of the variance is focused (over 80%!).")

# Show PCA result
st.markdown("**PCA Scatter Plot (PC1 vs PC2)**")
fig, ax = plt.subplots()

palette = {"Malignant": "blue", "Benign": "orange"}

for diagnosis in ["Malignant", "Benign"]:
    subset = pc[pc["Diagnosis"] == diagnosis]
    sns.scatterplot(data=subset, x="PC1", y="PC2", hue="Diagnosis", palette=palette, alpha=0.7, s=30)

ax.set(xlabel="PC1", ylabel="PC2", title="PCA Scatter Plot")
ax.legend()
st.pyplot(fig)


# K-Means clustering
st.subheader("K-Means Clustering")
st.markdown("Here you can see that K-means clustering has been applied with k=2 on the first two principle components (PC1 and PC2) of the breast cancer dataset. The model assigns each data point to one of two clusters, which we can visualise in a scatter plot. As this dataset includes labelled diagnoses (Malignant and Benign), we can compare the K-means clusters to the actual labels to assess how well the algorithm separates the two groups.")

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pc[['PC1', 'PC2']])
pc["Cluster"] = kmeans.labels_

# Show clustered data
st.markdown("**K-Means Clustering Scatter Plot (PC1 vs PC2)**")
fig, ax = plt.subplots()
for cluster in range(2):
    subset = pc[pc["Cluster"] == cluster]
    ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7, s=30)

# Plot cluster centers
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=100, c="black", label="Centroids")

ax.set(xlabel="PC1", ylabel="PC2", title=f"K-Means Clustering (k={num_clusters})")
ax.legend()
st.pyplot(fig)

# Check cluster accuracy
st.markdown("To measure the accuracy of the clustering, we can compare the assigned clusters with the actual diagnoses labels, Malignant = 0 and Benign = 1. While K-Means is not a supervised method, a strong alignment with true labels suggests that the data naturally separates into two distinct groups. However, if misclassification is high it may indicate an overlap in features, such as Malignant and Benign cases have similar characteristics in PC1 and PC2, suggesting a need for more features.")

st.markdown("**Confusion Matrix of Clustering Accuracy**")
st.markdown("""
- Top-left (0.79): 79% of Malignant cases were correctly clustered. 
- Top-right (0.21): 21% of Malignant cases were misclassified as Benign.
- Bottom-left (0.017): Only 1.7% of Benign cases were misclassified as Malignant.
- Bottom-right (0.98): 98% of Benign cases were correctly clustered. 

The clustering is aligned with the true labels. Most Benign cases are correctly classified (98%), with minimal misclassification. Malignant cases are also well-classified (79%), but there is still some overlap.""""")

bc_true = pc["Diagnosis"].map({"Malignant": 0, "Benign": 1})
bc_cluster = pc["Cluster"]

bc_conf_mat = confusion_matrix(bc_true, bc_cluster, normalize='true')

fig, ax = plt.subplots()
bc_plot = ConfusionMatrixDisplay(confusion_matrix=bc_conf_mat, display_labels=['Malignant', 'Benign'])
bc_plot.plot(ax=ax)
st.pyplot(fig)





 













