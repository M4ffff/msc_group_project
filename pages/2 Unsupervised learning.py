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
from Modules.unsupervised_functions import run_kmeans_animation, plot_clusters


st.title("Unsupervised Learning Page")


tab1, tab2, tab3 = st.tabs(["Introduction", "Unsupervised explanation", "Clustering Examples"])

with tab1:
    st.subheader("Introduction to Unsupervised Learning")
    
    st.markdown("Unsupervised learning is a type of machine learning where the algorithm learns patterns from unlabelled data,\
        uncovering hidden structures without predefined categories. This approach is particularly useful for exploring large datasets\
        and discovering relationships or groupings within the data.")
        
    st.subheader("Types of Unsupervised Learning")
    
    types_multi = """
    Unsupervised learning consists of techniques such as ***dimensionality reduction*** and ***clustering***.
    
    But what do these mean?
    
    """
    
    st.write(types_multi)
    
    with st.expander("Dimensionality Reduction"):
        st.write("""
                 This is a method to reduce the number of features in a dataset, without losing lots of important information. 
                 Often, datasets have many features. However, as humans we are unable to visualise data with more than 3 dimensions. 
                 Dimensionality reduction allows us to reduce the number of dimensions to a number that can be visualised by us!
                 
                 
                 The most important features of the dataset can also be determined, allowing future research to focus on these specific features and ignore irrelevant features.   
                 
                 There are both linear algorithms (eg principal component analysis (PCA)) and non-linear methods (eg t-SNE). 
                 
                 """)
        
    with st.expander("Clustering"):
        st.write("""
                 This is a method often used alongside dimensionality reduction.
                 It is used to group unlabelled data based on data points' similarity to each other.
                 
                 The similarity of the datapoints can be determined in different ways.
                 Some methods such as **K-means clustering** split the data into group purely based on distance to each other. 
                 Other methods such as DBSCAN, base simlarity on the density of data. 
                 
                 The method you want to use depends on each case. 
                 
                 """)
    
    st.write("Flick through the tabs at the top to explore these methods in much more detail!")
    
    st.subheader("Uses of unsupervised learning")

    uses_multi = """
    Unsupervised learning is particularly useful in data preparation and visualisation. 
    Often, data scientists do not know what they're looking for in a set of data. 
    Unsupervised learning allows them to determiine the most important features. 
    Can also determined unknown patterns and relationships. 
    
    Also reduces the chance of human error. 
    
    - Anomaly detection
    - Data preparation for supervised learning
    - Recommendation systems
    
    """
    st.write(uses_multi)
    
    
# Interactive graphs of the pre/post data for each marker. 
with tab2:
        
    st.subheader("A Real-Life Example!")
    
    st.write("Throughout the rest of this page, we will go through an example of using unsupervised techniques in a real-life example!")

    st.write("The following dataset is a record of multiple medical measurements of breast cancer tumours. ")
    st.write("Hopefully, using our clever unsupervised machine learning techniques, we can use this dataset to find a reliable way of diagnosing breast cancer!")
    st.write("This is an example that shows the usefulness of machine learning, and its possibility to even save lives! :female-doctor:")
    
    bc_dat = pd.read_csv('datasets/breast-cancer.csv')
    scaled_bc = StandardScaler().fit_transform(bc_dat[bc_dat.columns[1:]])

    st.dataframe(bc_dat.head())
    st.markdown("The columns, or ***features***, of the breast cancer dataset represent the variables measured for each data point. Whereas the data points themselves represent individual samples, with each row in the dataset being a different sample.")

    st.write("Because this page is demonstrating **unsupervised** learning, we are going to **drop** the diagnosis labelfrom the dataset. After we've used our unsupervised techniques, we will come back to these labels and see how our analysis did! ")

    st.markdown("Have a :eyes: at how the different features of the dataset interact! Can you find any features which are strongly related to another feature?")


    # Show original data (numerical columns)
    numeric_columns = bc_dat.select_dtypes(include=['number']).columns.tolist()
    x_axis = st.selectbox("Select x-axis:", numeric_columns)
    y_axis = st.selectbox("Selection y-axis:", numeric_columns)

    fig, ax = plt.subplots()
    sns.scatterplot(data=bc_dat, x=x_axis, y=y_axis, ax=ax, alpha=0.7)
    ax.set(xlabel=x_axis, ylabel=y_axis, title=f"{x_axis} vs {y_axis}")
    st.pyplot(fig)


    st.write("Hopefully with a bit of exploration, you've found some features that seem to correlate with each other!\
        If not, it may be worth going back to have another look...\
        However, if you did, you can click the checkbox below to see if we found the same pair of highly correlated features!" )
    
    if st.checkbox('Show highly correlated features'):
        feature1 = "*radius*"
        feature2 = "*perimeter*"
    else:
        feature1 = "??????"
        feature2 = "?????? "

        
    st.markdown(f"""
    
    But why does it matter if two features correlate?
    Well, if two features are strongly related, there isn't much point in having both in the analysis.
    For example, {feature1} and {feature2} are very strongly related.  

    This means if we measure the {feature1}, we don't also need to take a measurement for the {feature2}, as the {feature2} can be determined pretty accurately from the {feature1}.  
    This is how the number of features (aka the dimensions) can be reduced.
    """)



    st.subheader("Unsupervised techniques")
    
    st.write("In the analysis of this dataset, we will use Principal Component Analysis (PCA) and K-means clustering. ")
    
    
    st.write("PCA will simplify the dataset by reducing the number of features, while retaining as much of the important information from the data as possible.")
    with st.expander("PCA Process"):
        st.markdown('''
        Step 1: The data is standardised, preventing certain features from dominating the variance due to purely having a large magnitude. 
        
        Step 2: *Principal components* (a.k.a. new axes) are determined which attempt to maximise the amount of variance explained. these components are **linear combinations** of the original features.
        
        Step 3: The new prinicipal components are sorted by how much variance of the dataset they describe.
        Only the top few PC axes are kept, retaining the maximum amount of information in the fewest possible number of features.
        
        ***PCA is usually followed by a clustering algorithm.***
        ''')
        
    st.write("K-means clustering is a common way to group data into different categories based on how similar the data points are.")
    with st.expander("K-Means Process"):
        st.markdown('''
        Step 1: The number of groups, *k*, to cluster into is picked, with *k* data points then randomly selected to be the starting points for the cluster centres.  
        
        Step 2: The other data points are assigned to their nearest cluster centre.
        
        Step 3: Cluster centres are updated based on the average of the members of their group.
        
        ***This is repeated until the cluster centre positions are stable.***
        ''')

    st.markdown("**:rainbow[Example:]** Explore PCA and K-means clustering on the dataset below!")


    # Apply PCA
    st.subheader("After PCA")
    pca = PCA()
    transformed_bc = pca.fit_transform(scaled_bc)
    pc = pd.DataFrame(transformed_bc, columns=['PC{}'.format(i + 1) for i in range(transformed_bc.shape[1])])
    st.dataframe(pc.head())
    pc['Diagnosis'] = bc_dat['Diagnosis']

    st.markdown("PCA has transformed the original dataset into a new set of axes, known as principle components. The 1st principle component (PC1) captures the greatest variance in the data, the 2nd principle component (PC2) captures the second greatest variance and so on, capturing less and less variance for each principle component.")

    # Check variance ratio
    pc_var = pca.fit(scaled_bc)
    fig, ax = plt.subplots()
    ax.bar(pc.columns[:-1], height=pc_var.explained_variance_ratio_, label="Individual EV")
    ax.plot(pc.columns[:-1], np.cumsum(pc_var.explained_variance_ratio_), linestyle="--", label="Cumulative EV")
    ax.set(xlabel="Number of Components", ylabel="Cumulative Explained Variance")
    ax.legend()
    st.pyplot(fig)

    st.markdown("For this example we'll focus on the first two principle components, as the majority of the variance of the dataset is covered (over 80%!:astonished:).")


    pcs_relevance = ['PC1', 'PC2']
    num_pcs = len(pcs_relevance)
    feature_relevance_matrix = pd.DataFrame(pca.components_[:num_pcs].T * np.sqrt(pca.explained_variance_[:num_pcs]), 
                                columns=pcs_relevance, index=bc_dat.columns[1:])


    fig, ax = plt.subplots()
    for i in range(num_pcs):
        if i == 0:
            width=0.2
        else:
            width = -0.2
        ax.bar(bc_dat.columns[1:], feature_relevance_matrix[feature_relevance_matrix.columns[i]], width=width, align='edge', label=f'{pcs_relevance[i]}')
    ax.set_xlabel('Features')
    ax.set_ylabel('Relevance')
    plt.xticks(rotation=90)
    ax.legend()
    st.pyplot(fig)

    ######################## NEED TO DO THIS
    # st.write("This bar chart shows that texture and fractal dimension have the least variance explained by")

    # Show PCA result
    st.markdown("So, lets see what it looks like if we plot Principal Component 1 against Principal Component 2!")
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=pc, x="PC1", y="PC2", alpha=0.7, s=30)
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA Scatter Plot")
    # ax.legend()
    st.pyplot(fig)

    st.write("It looks like there may be two groups forming - lets try clustering to put them into two clusters - maybe they'll relate to benign/malignant diagnoses....")


    # K-Means clustering
    st.subheader("K-Means Clustering")
    st.markdown("As we know that there are two key groupings in the data, Malignant and Benign, therefore K-means clustering will be applied with two clusters (k=2). This ensures that the model assigns each data point to one of two clusters.")


    # Show clustered data
    st.markdown("Below we see the two clusters clearly defined. ")
    
    num_clusters = 2
    plot_clusters(pc, num_clusters=num_clusters)

    st.write("So, we've managed to produce two clusters. But how did this happen? Is there some hidden wizard who uses sorcery to determine the clusters for us? :magic_wand:")
    st.write("**No.**")
    st.write("Its the **K-Means clustering algorithm**. If you want to see how it works step-by-step, open the expander below for a more in-depth explanation \
             including an :rainbow[animation]. ")
    
    
    
    with st.expander("Detailed K-Means Clustering Process"):

        st.write("Click the big button :large_red_square::large_red_square: below to see how K-Means clustering determines the final clusters, \
            but first read about the process in more detail......")
        
        st.write("**The Process**")
        st.write(f"1. Select the number of clusters (in this case **{num_clusters}**).")
        st.write(f"2. Randomly select **{num_clusters}** points of data as starting points for the **{num_clusters}** cluster centres.")
        st.write("3. Calculate which centre each point is closest to.")
        st.write(f"4. Sort into the {num_clusters} clusters by which centre is closest.  :man-boy-boy:             :woman-girl-girl:")
        st.write("5. Calculate the mean of each cluster, with the mean then set as the new cluster centre. :abacus:")
        st.write("6. Repeat this process until the cluster centres stabilise, calculated with a given **tolerance** (in the animation, set as 0.1).")
        st.write("FINISHED :trophy:")

        run_kmeans_animation(transformed_bc[:, :2])



    st.write("But how do our clusters compare to the breast cancer diagnosis?? Do they correspond to benign/malignant tumours? Lets find out :point_right:")
    

    st.subheader("Clustering comparison")
    # Show PCA result
    st.markdown("**PCA Scatter Plot (PC1 vs PC2)**")
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    palette = {"Malignant": "blue", "Benign": "orange"}

    for diagnosis in ["Malignant", "Benign"]:
        subset = pc[pc["Diagnosis"] == diagnosis]
        sns.scatterplot(data=subset, x="PC1", y="PC2", hue="Diagnosis", palette=palette, alpha=0.7, s=30)

    ax[1].set(xlabel="PC1", ylabel="PC2", title="PCA Scatter Plot")
    ax[1].legend()
    
    plot_clusters(pc, fig, ax[0], num_clusters = 2)
    
    
    st.write("It can clearly be seen that the clusters determined by K-means clustering are very similar to the clusters produced if labelling by diagnoses.")
    st.write("This suggests that a patient could have a breast cancer tumour effectively diagnosed as benign or malignant, \
        by having these measurements taken and then seeing which cluster they would fall into after applying PCA.  ")
    
    st.write("But how accurate would the diagnosis be? All be explained below :point_down:")
    
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



with tab3:
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

    # st.image("images/2_unsupervised_images/funky_shapes.png", caption="Data with funky shapes", width=600)
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




