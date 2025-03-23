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
from sklearn.preprocessing import StandardScaler
import time


from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Unsupervised Learning Page")


tab1, tab2 = st.tabs(["Unsupervised explanation", "Clustering Examples"])



# Interactive graphs of the pre/post data for each marker. 
with tab1:
    st.markdown("Unsupervised learning is a type of machine learning where the algorithm learns patterns from unlabelled data, uncovering hidden structures without predefined categories. This approach is particularly useful for exploring large datasets and discovering relationships or groupings within the data.")
        



    st.subheader("PCA and K-means Clustering")
    st.write("PCA simplifies complex datasets by reducing the number of features, while keeping as much of the important information so that the significancy of the data is not affected.")
    with st.expander("PCA Process"):
        st.markdown('''
        Step 1: The data is standardised, so that all features are on the same scale. 
        
        Step 2: Key features are identified, through the combination of original features.
        
        Step 3: The dimensions are reduced, only the top few features are kept which retains the most significant information.
        
        ***PCA is usually followed by a clustering algorithm.***
        ''')
        
    st.write("K-means clustering is a common way to group data into different categories based on how similar the data points are.")
    with st. expander("K-Means Process"):
        st.markdown('''
        Step 1: The number of groups are picked, with random group centres. 
        
        Step 2: The data points are assigned to the nearest group.
        
        Step 3: Group centres are updated based on new groups.
        
        ***This is repeated until the best grouping is found.***
        ''')

    st.markdown("**:rainbow[Example:]** Explore PCA and K-means clustering on the dataset below!")

    st.subheader("Raw Breast Cancer Dataset")

    st.write("The following dataset is a record of multiple medical measurements of breast cancer tumours. ")
    st.write("Hopefully, using our clever unsupervised machine learning techniques, we can use this dataset to find a reliable way of diagnosing breast cancer!")
    st.write("This is an example that shows the usefulness of machine learning, and its possibility to even save lives! :female-doctor:")
    
    bc_dat = pd.read_csv('datasets/breast-cancer.csv')
    scaled_bc = StandardScaler().fit_transform(bc_dat[bc_dat.columns[1:]])

    st.dataframe(bc_dat.head())
    st.markdown("The columns, or ***features***, of the breast cancer dataset represent the variables measured for each data point. Whereas the data points themselves represent individual samples, with each row in the dataset being a different sample.")

    st.markdown("Have a :eyes: at how the different features of the dataset interact!")


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

    st.markdown("PCA has transformed the original dataset into a new set of axes, known as principle components. The 1st principle component (PC1) captures the greatest variance in the data, the 2nd principle component (PC2) captures the second greatest variance and so on, capturing less and less variance for each principle component.")

    # Check variance ratio
    pc_var = pca.fit(scaled_bc)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pc_var.explained_variance_ratio_))
    ax.set(xlabel="Number of Components", ylabel="Cumulative Explained Variance")
    st.pyplot(fig)

    st.markdown("For this example we'll focus on the first two principle components, as the majority of the variance of the dataset is covered (over 80%!:astonished:).")

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
    st.markdown("As we know that there are two key groupings in the data, Malignant and Benign, K-means clustering will be applied with two clusters (k=2). This ensures that the model assigns each data point to one of two clusters.")

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


    st.title("K-Means Clustering Animation")

    st.write("Click the big button below to see how K-Means clustering determines the final clusters......")
    st.write("Also press the expander to read about the K-Means clustering process in more detail.")

    with st.expander("Detailed K-Means Clustering Process"):
        st.write("**The Process**")
        st.write("1. Select the number of clusters (in this case **two**).")
        st.write("2. Randomly select **two** points of data as starting points for the **two** cluster centres.")
        st.write("3. Calculate which centre each point is closest to.")
        st.write("4. Sort into the two clusters by which centre is closest.  :man-boy-boy:             :woman-girl-girl:")
        st.write("5. Calculate the mean of each cluster, with the mean then set as the new cluster centre. :abacus:")
        st.write("6. Repeat this process until the cluster centres stabilise, calculated with a given **tolerance** (in the animation, set as 0.1).")
        st.write("FINISHED :trophy:")


    button_placeholder = st.empty()

    # Streamlit app
    fig, ax = plt.subplots(figsize=(6, 4))

    X = transformed_bc[:, :2]
    n_clusters=2

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

    # loop through number of iterations
    if button_placeholder.button("Run animation", type="primary"):
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


    st.subheader("Final Analysis")

    st.write("Here, we run through how to anlayse our ML methods, and see if our method is an accurate way of diagnosing breast cancer.")
    st.write("We sure hope so! :crossed_fingers:")


    # Check cluster accuracy
    st.write("As this dataset includes labelled diagnoses, Malignant and Benign, we can compare the K-means clusters to the actual labels (Malignant = 0 and Benign = 1) to assess how well the algorithm separates the two groups.")
    with st.expander("Why is this useful? :thinking_face:"):
        st.markdown('''
        This is beneficial as while K-Means is not a supervised method, a strong alignment with true labels suggests that the data naturally separates into two distinct groups.  
        However, if misclassification is high it may indicate an overlap in features, such as Malignant and Benign cases having similar characteristics in PC1 and PC2, potentially suggesting the need for more features.''')

    # Confusion matrix
    st.markdown("**Confusion Matrix of Clustering Accuracy**")
    bc_true = pc["Diagnosis"].map({"Malignant": 0, "Benign": 1})
    bc_cluster = pc["Cluster"]

    bc_conf_mat = confusion_matrix(bc_true, bc_cluster, normalize='true')

    fig, ax = plt.subplots()
    bc_plot = ConfusionMatrixDisplay(confusion_matrix=bc_conf_mat, display_labels=['Malignant', 'Benign'])
    bc_plot.plot(ax=ax)
    st.pyplot(fig)

    st.markdown("""
    :green[Top-left]: 79% of Malignant cases were correctly clustered. 
    
    :blue[Top-right]: 21% of Malignant cases were misclassified as Benign.
    
    :violet[Bottom-left]: 1.7% of Benign cases were misclassified as Malignant.
    
    :orange[Bottom-right]: 98% of Benign cases were correctly clustered. 

    The K-means model was well aligned with the true labels. Most Benign cases were correctly classified (98%:astonished:), with minimal misclassification. Malignant cases were also well-classified (79%), but there is still some overlap.""""")

    st.markdown("""
    **:rainbow[Questions to think about!]**
    - Given the results of the confusion matrix, what does this mean for the model?
    - Could the model be used to diagnose real breast cancer cases?
    - Could we include more principle components to hinder the overlap?
    """)



with tab2:
    st.subheader("INTERACTIVE PLOT TO SEE HOW CLUSTERING WORKS WITH DIFFERENT DATASHAPES")


    # normal writing details
    multi = '''
    Data doesnt always come in blobs - it can come in *other shapes*:    
    Different clustering techniques work better than others depending on the shape of the data. :large_blue_square: :large_blue_diamond: :large_blue_circle: 

    The plot below allows you to produce some data selected randomly from a set of :rainbow[funky] datasets -  
    For example there is data in the shape of smiley faces :smiley:, dartboards :dart:, yin yang :yin_yang:, spirals :face_with_spiral_eyes:, and so much more!

    I hope that this will show you how clustering works for different data shapes in a *fun* :rainbow[interactive] way,\
        with the benefits and limitations of the different methods made clearer as you cluster different shapes using each method. 

    **So, what clustering methods are available?**
    
    The following methods are the four main methods for clustering, (although there are others out there!):
    - :orange[K-means clustering]
    - :violet[Gaussian Mixture Model (GMM) clustering]
    - :green[DBSCAN]
    - :red[HDBSCAN]

    To understand how these different methods work, click on the expander tabs below....

    '''
    st.markdown(multi)

    # st.image("images/funky_shapes.png", caption="Data with funky shapes", width=600)
    with st.expander("**:orange[K-Means clustering]:**"):
        st.markdown(
            """
        Hopefully, you saw this in action on the page before, but we'll run through it again in case you didn't!
        
        1. Select the number of clusters, for example **three**.
        2. Randomly select **three** points of data as starting points for the **three** cluster centres.
        3. Calculate which centre each point is closest to.")
        4. Sort into the three clusters by which centre is closest.  :man-boy-boy:             :woman-girl-girl:
        5. Calculate the mean of each cluster, with the mean then set as the new cluster centre. :abacus:
        6. Repeat this process until the cluster centres stabilise, calculated with a given **tolerance**.
        "FINISHED :trophy:
        """)
        
    with st.expander("**:violet[GMM clustering]:**"):
        st.markdown(
            """
        GMM assumes the data is a mixture of multiple Gaussian distributions, each corresponding to a cluster.
        The number of clusters must be specified, as with K-means clustering.
        Gmm iteratively estimates the parameters of the Gaussian distributions, using the \"Expectation-Maximisation algorithm\"
        The data points are each given a probability of belonging in each cluster.
        
        Gmm can deal with overlapping clusters, and more funky-shaped data. 
        """)
        
    with st.expander("**:green[DBSCAN clustering]:**"):
        st.markdown(
            """
        DBSCAN clusters data based on their density. 
        Clusters are areas of high density separated by areas of low density.
        Clusters are identified by determining points with a high number of other points in close proximity,
        and expanding clusters from these points. 
        DBSCAN does not need the number of clusters specified, but does require sensible input parameters. 
        These parameters are as follows:
        - **Eps**:  *The maximum distance between two samples for one to be considered as in the neighbourhood of the other.*
        - **min_clusters**: *Minimum number of points for a group of data to be considered a cluster rather than noise.*
        
        DBSCAN does not assume the shapes of the clusters, so is often more effective at clustering strangely-shaped data, 
        which form obvious clusters to the human eye, but which K-means and GMM may struggle with. 
        """)
        
    with st.expander("**:red[HDBSCAN clustering]:**"):
        st.markdown(
            """
        HDBSCAN is an extension of DBSCAN, with can deal effectively with clusters of varying densities. 
        The "H" stands for "hierarchical". This is because it creates a hierarchy (ranking) of  clusters based on their density.
        This is done automatically, although the technique could be improved by tweaking some input parameters. 
        """)

    final_multi = """    These different methods often produce different clusters on the same set of data, as we will see below./
    Therefore, its up to **:rainbow[you]** to determine which method (or methods) outputs the correct clusters!
"""
    st.write(final_multi)


    #######################################################################################################################################
    st.subheader("LOADING IN FUNKY DATA")

    # Making filename dictionary
    
    # SHOULD MOVE OUT OF SCRIPT
    cluster_dict = {
    "basic1.csv": {"best_method": ["GMM"], "num_clusters": 4,
                   "description": "This data is split into 4 blobs which GMM clusters effectively."},
    "basic2.csv": {"best_method": ["DBSCAN"], "num_clusters": 5,
                   "description": "DBSCAN definitely the most effective here (eps=15,min_clusters=5) - the other methods struggle with the elongated shapes of the clusters."},
    "basic3.csv": {"best_method": ["GMM"], "num_clusters": 3,
                   "description": ""},
    "basic4.csv": {"best_method": ["K-means", "GMM"], "num_clusters": 3,
                   "description": "K-means and GMM are most effective here due to the blobby nature of the data."},
    "basic5.csv": {"best_method": ["K-means", "GMM"], "num_clusters": 3,
                   "description": "K-means and GMM both effectively cluster here. DBSCSN and HDBSCAN struggle with the sparsity of the data."},
    "blob.csv": {"best_method": ["K-means", "GMM"], "num_clusters": 4,
                   "description": "These clusters are not clearly distinct, but there is a vague blobby structure which K-means and GMM show."},
    # "box.csv": {"best_method": "dbscan", "num_clusters": 1}, ###
    # "boxes.csv": {"best_method": "dbscan", "num_clusters": 30}, ###
    "boxes2.csv": {"best_method": ["DBSCAN"], "num_clusters": 3,
                   "description": "All the methods struggle with this one,  particularly as noise increases. However, with low noise, DBSCAN determines the three clusters most reliabely."},
    # "boxes3.csv": {"best_method": "dbscan", "num_clusters": 12}, ###
    "chrome.csv": {"best_method": ["DBSCAN", "HDBSCAN"], "num_clusters": 4,
                   "description": "DBSCAN (eps=0.32, min_clusters=5) and HDBSCAN both effective here due to the strange shapes of the clusters,\
                       even as noise increases."},
    "dart.csv": {"best_method": ["DBSCAN", "HDBSCAN"], "num_clusters": 2,
                   "description": "K-means particularly struggles with this one as the clusters do not have blobby shapes. "},
    "dart2.csv": {"best_method": ["DBSCAN", "HDBSCAN"], "num_clusters": 4,
                   "description": "DBSCAN and HDBSCAN can effectively cluster based on the shapes of the rings (although struggle when the noise is increased) "},
    "face.csv": {"best_method": ["DBSCAN", "HDBSCAN"], "num_clusters": 4,
                 "description": "DBSCAN (eps=0.32, min_clusters=5) and HDBSCAN both effective here, even as noise increases."},
    "hyperplane.csv": {"best_method": ["GMM"], "num_clusters": 2,
                   "description": "GMM probably the most effective here, although there aren't any clearly defined clusters so hard to judge the effectiveness of each method."},
    "isolation.csv": {"best_method": ["DBSCAN"], "num_clusters": 3,
                   "description": ""},
    "lines.csv": {"best_method": ["DBSCAN"], "num_clusters": 5,
                   "description": ""},
    "lines2.csv": {"best_method": ["DBSCAN"], "num_clusters": 5,
                   "description": "DBSCAN the most effective here (eps=?,min_clusters=?) - the other methods struggle with the elongated shapes of the clusters."},
    "moon_blobs.csv": {"best_method": ["DBSCAN"], "num_clusters": 4,
                   "description": ""},             ############## check best method
    "network.csv": {"best_method": ["K-means", "GMM", "DBSCAN"], "num_clusters": 5,
                   "description": "All the methods work pretty effective here, although GMM deals with increased noise the best. DBSCAN:(eps=0.10,min_clusters=5)"},
    "outliers.csv": {"best_method": ["GMM"], "num_clusters": 2,
                   "description": ""},
    # "ring.csv": {"best_method": "kmeans", "num_clusters": 1}, ###
    "sparse.csv": {"best_method": ["K-means", "GMM"], "num_clusters": 3,
                   "description": "This is pretty blobby data, so K-means and GMM both effectively cluster into 3 clusters. DBSCAN struggles with the sparsity of the data here"},
    "spiral.csv": {"best_method": ["DBSCAN"], "num_clusters": 1,
                   "description": ""}, ###
    "spiral2.csv": {"best_method": ["GMM"], "num_clusters": 2,
                   "description": ""},
    "spirals.csv": {"best_method": ["GMM"], "num_clusters": 3,
                   "description": ""},
    "supernova.csv": {"best_method": ["GMM"], "num_clusters": 4,
                   "description": ""},
    "triangle.csv": {"best_method": ["DBSCAN"], "num_clusters": 3,
                   "description": ""},
    "un.csv": {"best_method": ["GMM"], "num_clusters": 2,
                   "description": ""},
    "un2.csv": {"best_method": ["DBSCAN"], "num_clusters": 3,
                   "description": ""},
    "wave.csv": {"best_method": ["GMM"], "num_clusters": 4,
                   "description": ""}
}



    ################################################## 

    def get_data(seed):
        random.seed(seed)
        random_file = random.choice(list(cluster_dict.keys())) 
        # random_file = (list(cluster_dict.keys()))[6]
        # st.write(random_file)

        random_data = pd.read_csv(f"datasets/cluster data/{random_file}")
        
        # drop color column if it exists
        random_data = random_data.drop(columns=["color"], errors='ignore')
        
        # normalise data    
        random_data = pd.DataFrame( StandardScaler().fit_transform(random_data.values), columns=random_data.columns )
                
        return random_file, random_data

    
    # determine random starting seed of session if first run
    if "initial_seed" not in st.session_state:
        initial_seed = random.randint(0, 100)
        st.write(f"Initial seed counter: {initial_seed}")
        st.session_state["initial_seed"] = initial_seed
    
    # make session counter if it doesnt exist, and set equal to initial seed
    if "counter" not in st.session_state:
        st.session_state["counter"] = st.session_state["initial_seed"]
    
        
    session_counter = st.session_state["counter"]
    # st.write(f"Session seed counter: {session_counter}")


    random_file, random_data = get_data(st.session_state["counter"])

    # Select and plot random file of funky data
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.button("Get new data", type="primary"):
            
            st.session_state["counter"] += 1
            random_file, random_data = get_data(st.session_state["counter"])
    with col2:
        if st.session_state["counter"] > st.session_state["initial_seed"]:
            if st.button("Review previous data", type="secondary"):
                
                st.session_state["counter"] -= 1
                random_file, random_data = get_data(st.session_state["counter"])



    st.write("Make the lil quiz below **harder** by increasing the noise of the data.\
             This will make the clusters less distinct. (and may make the models significantly worse at determining the clusters) ")
    noise_std = st.slider("Select noise distribution", 0.0, 0.5, step=0.1 )
    rand_noise = np.random.normal(0,noise_std, random_data.shape )

    random_data += rand_noise


    #######################################################################################################################################
    st.subheader("CLUSTER AND PLOT FUNKY DATA")

    # have plot next to explanation of limitations of each model. 

    col1, col2 = st.columns([0.7, 0.3])


    def basic_plot(data, ax, size=10, colour_labels=np.zeros(len(random_data)), cluster_centres=None):
        """
        Basic scatter plot of some data
        
        """
        
        ax.clear()
        import matplotlib.cm as cm
        
        # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        # for y, c in zip(ys, colors):
        #     plt.scatter(x, y, color=c)
        
        ax.scatter(data["x"], data["y"], c=colour_labels, s=size, cmap="gist_rainbow")
        ax.set_aspect("equal")
        
        if isinstance( cluster_centres, np.ndarray):
            ax.scatter(cluster_centres[:, 0], cluster_centres[:, 1], s=size, marker='s', c="cyan")

        # st.pyplot(fig)

        
    def kmeans_cluster(data, num_centres):
        kmeans = KMeans(n_clusters=num_centres)
        kmeans.fit(data)
        
        labels = kmeans.labels_
        cluster_centres = kmeans.cluster_centers_
        
        return labels, cluster_centres

    def gmm_cluster(data, num_centres):        
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=num_centres).fit(data)
        colour_labels = gmm.predict(data)
        
        probabilities = gmm.predict_proba(data)
        size = 15 * probabilities.max(axis=1) ** 2
        
        return size, colour_labels


    def dbscan_cluster(data, eps, min_samples):
        from sklearn.cluster import DBSCAN
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = dbscan.labels_
        
        num_labels = len(np.unique(labels))
        st.write(f"Number of unique labels: {num_labels}")

        return labels


    # not sure if this works
    def hdbscan_cluster(data):
        from sklearn.cluster import HDBSCAN
        
        labels = HDBSCAN().fit_predict(data)
        # labels = hdbscan.labels_
        
        # num_labels = len(np.unique(labels))
        # st.write(f"Number of unique labels: {num_labels}")

        return labels

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,6))
        
        
        basic_plot(random_data, ax1)
        # st.pyplot(fig1)

    with col2:
        clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN", "HDBSCAN"]
        technique = st.selectbox('Select a technique for clustering.', clustering_technique_options)
            
        cluster_centres = None
        if technique == clustering_technique_options[1]:
            num_clusters = st.slider("Select number of clusters", 1, 6, value=3)
            colour_labels, cluster_centres = kmeans_cluster(random_data, num_clusters)
            size = 15
            
        elif technique == clustering_technique_options[2]:
            num_clusters = st.slider("Select number of clusters", 1, 6, value=3)
            size, colour_labels = gmm_cluster(random_data, num_clusters)
            
        elif technique == clustering_technique_options[3]:
            st.write("Change the following slider to see how this parameter effects the DBSCAN results.")
            eps = st.slider("Eps", 0.01, 0.5, value=0.10, step=0.01)
            min_clusters = st.slider("min_clusters", 1, 15, value=5)
            
            colour_labels = dbscan_cluster(random_data, eps, min_clusters)
            size=15
            
        elif technique == clustering_technique_options[4]:
            st.write("hdbscan")
            # eps = st.slider("Eps", 1, 15)
            colour_labels = hdbscan_cluster(random_data)
            size=15
            
        else:
            size = 15
            colour_labels = np.zeros(len(random_data))
            
    if technique == clustering_technique_options[3]:
        if st.checkbox("Click to see what 'Eps' is....."):
            st.write("Eps is a DBSCAN-SPECIFIC DEFINITION: :nerd_face:")
            st.markdown("**Eps:** *The maximum distance between two samples for one to be considered as in the neighbourhood of the other.*")
            # st.markdown("<br>---------------------", unsafe_allow_html=True)
            st.markdown("***")
        
    # st.write("previous data slider bool:", st.session_state["previous_data_slider"])
        st.write("counter:", st.session_state["counter"])
    

    with col1:
        basic_plot(random_data, ax1, size, colour_labels, cluster_centres)
        st.pyplot(fig1)
        
    st.write(f"filename: {random_file}")   

    
    st.subheader("Quiz time!")
    # Quiz.
    col1, col2 = st.columns(2)
    with col1:
        question_one = st.radio(
            "How many clusters would you say there are?",(np.arange(1,7)), index=None)
        
    if question_one == cluster_dict[random_file]["num_clusters"]:
        col1.success("I agree with you :smile:!")
        
        with col2:
            question_two = st.radio(
            "Now, what method is best in determining this?",
            ("K-means", "GMM", "DBSCAN", "HDBSCAN"), index=None)
            
            if question_two == None:
                st.write("") 
                
            elif question_two in cluster_dict[random_file]["best_method"] and len(cluster_dict[random_file]["best_method"]) == 1 :
                st.success("Yeah! Was there ever any doubt? :smirk:")
                                
            elif question_two in cluster_dict[random_file]["best_method"] and len(cluster_dict[random_file]["best_method"]) > 1 :
                st.success("Yeah, I agree! However, other options may be just as good? :eyes:")
                                
            else:
                # st.write(f"clusters: {cluster_dict[random_file]["num_clusters"]}")
                st.error("I'm afraid I don't agree with you here, have another go!")
                    
    # other methods
    # elif question_one == cluster_dict[random_file]["cluster"]:
    #     print("This wouldn't be my first choice but it works well enough! ")
    elif question_one == None:
        st.write("")    
    else:
        st.error("I'm afraid I don't agree with you here, have another go!")
    
    
    if "description" in cluster_dict[random_file].keys():
            st.write(cluster_dict[random_file]["description"])
   



    st.subheader("CLUSTER SELECTION")


    end_multi = '''
    From this exercise, I hope you now feel like you understand the different clustering techniques better!  
    There is evidently no one-size-fits-all :gloves: clustering technique which can be used to cluster any set of data -\
    the methods introduced here all have their pros and cons. 
    
    Overall, GMM is generally the most reliable (and easy to implement), in particular for noise-less :mute: data. 
    However, it does struggle with clusters that are funky shapes if there is more noise and overlap between clusters. \
        In these cases, it often returns to clustering into blobs rather than effectively capturing the shape of the data.  
    
    It is important to remember that clusters are often hard to distinguish by eye. :eye: 
    This exercise uses fake data to allow easy comparison between the accuracy of the different methods in a more :rainbow[interesting] way. 
    However, unfortunately, it is rare for a dataset to have the shape of a smiley face in real-life datasets :disappointed: 
    
    To finish off, I'll give a little rundown of the pros and cons of each method: :white_check_mark: :x:
    
    Before checking what I've put, why don't you come up with your own pros and cons list for each method!
    
    '''

    st.markdown(end_multi)

    def pros_and_cons(multi_pros, multi_cons):
       
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
            
    
    kmeans_tab, gmm_tab, dbscan_tab, hdbscan_tab = st.tabs(["K-means", "GMM", "DBSCAN", "HDBSCAN"])
    
    with kmeans_tab:
        multi_pros = '''
        - Simple to implement  
        - Good with blobby data 
        - Efficient with large datasets
        '''
        multi_cons = '''
        - Assumes data is blobby - can't determine more interesting shapes of clusters
        - Clusters depend on starting points 
        - Sensitive to outliers
        '''
        pros_and_cons(multi_pros, multi_cons)
        
    with gmm_tab:
        multi_pros = '''
        - Simple to implement  
        - Good with blobby data 
        - Good with interestingly-shaped clusters (but only if distinct)
        - Shows probability of each point being in its designated cluster
        - Handles overlap of clusters
        '''
        multi_cons = '''
        - Number of clusters must be determined before 
        - Bad with intertwined data
        - Assumes data has a Gaussian distribution
        - Sensitive to initial parameters
        '''
        pros_and_cons(multi_pros, multi_cons)
        
    with dbscan_tab:
        multi_pros = ''' 
        - Good with strangely shaped clusters 
        - Shows which points are likely noise
        - Number of clusters does not need to be pre-determined
        '''
        multi_cons = '''
        - Highly dependent on parameters - must be chosen carefully
        - Can produce an extremely larger number of clusters
        - Struggles with clusters of varying density
        '''
        pros_and_cons(multi_pros, multi_cons)


    with hdbscan_tab:
        multi_pros = '''
        - No parameters required 
        - Good with strangely shaped clusters 
        '''
        multi_cons = '''
        - Computationally expensive
        '''
        pros_and_cons(multi_pros, multi_cons)


    conclusion_multi = """
    So, there we have it - **UNSUPERVISED LEARNING**!
    
    I hope you feel like you've learnt a lot along the way, and now understand the different scenarios for when to use the different clustering methods. 
    
    Make sure to checkout the documentation page for simple examples of how to implement these methods in Python, \
    or view our GitHub (url) to see the full code used in producing these examples. 
    
    Unsupervised learning, over and out. :end:
    """

    st.write(conclusion_multi)




