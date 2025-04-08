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

from Modules.unsupervised_functions import run_kmeans_animation, plot_clusters, get_data, basic_plot, pros_and_cons
from Modules.unsupervised_functions import cluster_dict, kmeans_cluster, gmm_cluster, dbscan_cluster, hdbscan_cluster

st.title("Unsupervised Learning Page")


tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Real-life Example", "Clustering Explorations", "Python Implementation"])

with tab1:
    st.subheader("What ***is*** Unsupervised Learning:question::exclamation:")
    
    st.markdown("Unsupervised learning is a type of machine learning where the algorithm learns patterns from ***unlabelled*** data. Unlike supervised learning, there are no predefined outputs. The goal is to explore the underlying structure of the dataset, uncovering hidden relationships or groupings without predefined categories.")
        
    st.subheader("Types of Unsupervised Learning")
    
    types_multi = """
    Two core techniques of unsupervised learning are ***dimensionality reduction*** and ***clustering***.
    
    But what do these mean?
    
    """
    
    st.write(types_multi)
    
    with st.expander(":blue[Dimensionality Reduction] :arrow_right: :arrow_left:"):
        st.write("""
                 This is a method to reduce the number of features in a dataset, without losing lots of important information. 
                 Often, datasets have many features. However, as humans we are unable to visualise data with more than 3 dimensions. 
                 Dimensionality reduction allows us to **reduce the number of dimensions** to a number that can be visualised by us (i.e. 2D, or 3D)!
                 It is also useful for reducing the size of large datasets, making analysis less computationally intensive. :fast_forward:
                 
                 The most important features of the dataset can also be determined, allowing future research to focus on these specific features and ignore irrelevant features.   
                 
                 There are both *linear* algorithms (eg principal component analysis (PCA)) and *non-linear* methods (eg t-SNE). 
                 In this resource, we only really cover linear algorithms :disappointed: However, if you're interested in non-linear methods they're definitely worth looking into!
                 
                 """)
        
    with st.expander(":red[Clustering]"):
        st.write("""
                 This is a method often used alongside dimensionality reduction.
                 It is used to **group** unlabelled data based on the data points' similarity to each other.
                 
                 The similarity of the datapoints can be determined in different ways.
                 Some methods, such as **K-means clustering**, split the data into group purely based on distance to each other. 
                 Other methods, such as DBSCAN, base similarity on the density of data. 
                 
                 The method you want to use depends on each case - as we will explore in the next few tabs... 
                 
                 """)
    
    st.write("Flick through the tabs at the top to explore these methods in much more detail!")
    
    st.subheader("Unsupervised Learning applications")

    st.markdown("Unsupervised learning is particularly valuable in the *early stages* of data analysis, especially for data preparation and visualisation. It is often used when data scientists do not have predefined labels or clear hypotheses, allowing them to :rainbow[explore the data] freely.")
    st.markdown("By uncovering the most relevant features and revealing hidden patterns or relationships, unsupervised learning helps reduce the risk of human bias or oversight. Common applications include anomaly detection, preparing datasets for supervised learning, and powering recommendation systems by identifying natural groupings or preferences within the data.")
    
    
# Interactive graphs of the pre/post data for each marker. 
with tab2:
        
    st.subheader("A ***:rainbow[real-life]*** example!")
    
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

    fig, ax = plt.subplots(figsize=(6,4))
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



    st.subheader("Our Unsupervised approach")
    
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
    ax.plot(pc.columns[:-1], np.cumsum(pc_var.explained_variance_ratio_), color='blue', linestyle="--", label="Cumulative EV")
    ax.set(xlabel="Number of Components", ylabel="Explained Variance Ratio")
    ax.legend()
    st.pyplot(fig)

    st.markdown("For this example we'll focus on the first two principle components, as the majority of the variance of the dataset is covered (over 80%!:astonished:).")

    st.write("But which features are most important? Below we plot the weightings of each feature in the first two principal components.")
    
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

    
    st.write("This bar chart shows that each feature has a positive loading for PC1 which suggests a large PC1 corresponds to large quantities for the features.\
        The negative loadings for radius, texture, perimeter, and area for PC2, show that a larger value of PC2 corresponds to smaller size and texture.     ")
    st.write(" Furthermore, the texture and fractal dimension have the least variance explained by the first principal components.\
        This may be relevant later, so keep this in mind...")


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
    st.markdown("As we know that there are two key groupings in the data (Malignant and Benign), K-means clustering will be applied with two clusters (k=2). This ensures that the model assigns each data point to one of two clusters.")


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
        by having these measurements taken and then seeing which cluster they would fall into after applying PCA.\
            Furthermore, the clustering suggests that getting a larger PC1 score means the tumour is more likely to be malignant.\
                Associating this with the feature relevance bar chart above, texture and fractal dimension have the least relevance for diagnosis due to their small loadings of PC1. ")
    
    
    
    st.write("But how accurate would the diagnosis be? All be explained below :point_down:")
    
    st.subheader("Final Analysis")

    st.write("Here, we run through how to analyse our ML methods, and see if our method is an accurate way of diagnosing breast cancer.")
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
    st.subheader("How does clustering work on ***weird*** datasets:question::exclamation:")


    # normal writing details
    multi = '''
    Data doesnt always come in blobs - it can come in *other shapes*:    
    Different clustering techniques work better than others depending on the shape of the data.

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



    st.subheader("Cluster Experiment")

    # Making filename dictionary
    
    # SHOULD MOVE OUT OF SCRIPT


    
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


    random_file, random_data = get_data(st.session_state["counter"], cluster_dict)

    # Select and plot random file of funky data
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        if st.button("Get new data", type="primary"):
            
            st.session_state["counter"] += 1
            random_file, random_data = get_data(st.session_state["counter"], cluster_dict)
            
    # allows going to look back at previous data
    with col2:
        if st.session_state["counter"] > st.session_state["initial_seed"]:
            if st.button("Review previous data", type="secondary"):
                
                st.session_state["counter"] -= 1
                random_file, random_data = get_data(st.session_state["counter"], cluster_dict)



    # have plot next to explanation of limitations of each model. 

    col1, col2 = st.columns([0.7, 0.3])

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
    
    
    if st.checkbox("Show brief author analysis for this set of data!"):
        st.write(cluster_dict[random_file]["description"])
   



    st.subheader("CLUSTER SELECTION")


    end_multi = '''
    From this exercise, I hope you now feel like you understand the different clustering techniques better!  
    There is evidently no one-size-fits-all :gloves: clustering technique which can be used to cluster any set of data -\
    the methods introduced here all have their pros and cons. 
    
    Overall, DBSCAN is generally the most reliable method (and easy to implement), as it can deal with funky shapes and \
        noisy :mute: data. However, it requires tuning of the hyperparameters to ensure a correct clustering and can struggle when clusters overlap.
        GMM is easy to implement and deals with a lot of the datasets well.\
    However, it does struggle with clusters that are funky shapes especially if there is more noise. \
        In these cases, it often returns to clustering into blobs rather than effectively capturing the shape of the data.  
    
    It is important to remember that clusters are often hard to distinguish by eye. :eye: 
    This exercise uses fake data to allow easy comparison between the accuracy of the different methods in a more :rainbow[interesting] way. 
    However, unfortunately, it is rare for a dataset to have the shape of a smiley face in real-life datasets :disappointed: 
    
    To finish off, I'll give a little rundown of the pros and cons of each method: :white_check_mark: :x:
    
    Before checking what I've put, why don't you come up with your own pros and cons list for each method!
    
    '''

    st.markdown(end_multi)


            
    
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
        - Doesn't always capture clusters correctly.
        '''
        pros_and_cons(multi_pros, multi_cons)


    conclusion_multi = """
    So, there we have it - **UNSUPERVISED LEARNING**!
    
    I hope you feel like you've learnt a lot along the way, and now understand the different scenarios for when to use the different clustering methods. 
    
    Make sure to checkout the documentation page for simple examples of how to implement these methods in Python, \
    or view our GitHub (link on final page) to see the full code used in producing these examples. 
    
    Unsupervised learning, over and out. :end:
    """

    st.write(conclusion_multi)


with tab4:
    st.subheader("Python implementation")
    
    st.write("Here, we'll run through how to implement the tools used on this page in **Python**")
    st.write("If you have no experience in coding or just don't want to know how to implement these concepts in Python, that is *completely fine*, just skip on to the next page.\
        However, if you're an ***advanced learner*** crack on through this page.")
    st.write("Be warned, a basic understanding knowledge of Python is required to understand this page. ")
    
    
    st.subheader("Dimensionality reduction (PCA)")
    pca_multi = """
    To reduce the number of dimesnions of a dataset using PCA in Python the following steps are done. 
    
    First you need to scale the data, to ensure larger features do not dominate the variance, purely because of their magnitude. 
    ``` python
    from sklearn.preprocessing import StandardScaler

    scaled_data = StandardScaler().fit_transform(data)
    ```
    
    Next, we need to fit and transform the data. This can achieved in one function. 
    ``` python
    from sklearn.decomposition import PCA
    
    pca = PCA()
    transformed_data = pca.fit_transform(scaled_data)
    pc_var = pca.fit(scaled_data)
    ```
    This will have produced a new dataframe in which the columns represent each principal component. 
    
    There are multiple ways to get information from this transformed data. 
    
    ``` python
    
    # get proportion of total variance explained by each principal component
    pca.explained_variance_ratio_
    
    # Get feature components that make up each principal component
    pca.components_
    ```
    
    
    """
    st.markdown(pca_multi)
    
    st.subheader("Clustering")
    
    python_implementation_multi = """
    
    Here, I give a little function showing how to implement K-means clustering on a set of data.
    We use a module called ***sklearn*** to implement the clustering technique.
    This does all the hard work for you!
    
    ``` python
    from sklearn.cluster import KMeans
    
    def kmeans_cluster(data, num_centres):
        \"""
        Cluster data using kmeans clustering

        Args:
            data (df): input data to cluster
            num_centres (int): Number of clusters

        Returns:
            labels: labels for each datapoint of which cluster they are in
            cluster_centres: coordinates of cluster centres
        \"""
        kmeans = KMeans(n_clusters=num_centres)
        kmeans.fit(data)
        
        labels = kmeans.labels_
        cluster_centres = kmeans.cluster_centers_
        
        return labels, cluster_centres
    ```
    Notice how you must define the number of clusters. 
    
    The Gaussian Mixture Model is very similar:
    
    ``` python
    from sklearn.mixture import GaussianMixture
    
    def gmm_cluster(data, num_centres):     
        \"""
        Cluster data using gmm clustering

        Args:
            data (df): input data to cluster
            num_centres (int): Number of clusters

        Returns:
            size: Size proportional to probabilty of data being in given cluster
            colour_labels: labels for each datapoint of which cluster they are in
        \"""   

        gmm = GaussianMixture(n_components=num_centres).fit(data)
        colour_labels = gmm.predict(data)
        
        probabilities = gmm.predict_proba(data)
        size = 15 * probabilities.max(axis=1) ** 2
        
        return size, colour_labels
    
    ```
    
    The DBSCAN method is a little different, as you do not need to input the number of clusters, and can the hyperparameters 'eps' and 'min_samples'.
    
    ``` python
    from sklearn.cluster import DBSCAN
    
    def dbscan_cluster(data, eps, min_samples):
        \"""
        Cluster data using dbscan clustering

        Args:
            data (df): input data to cluster
            eps (int): Max distance for points to be neighbours.
            min_samples (int): Minimum number of points needed to form a cluster.

        Returns:
            labels: labels for each datapoint of which cluster they are in
        \"""   
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = dbscan.labels_
        
        num_labels = len(np.unique(labels))
        st.write(f"Number of unique labels: {num_labels}")

        return labels
    ```
    
    """

    st.write(python_implementation_multi)
    
