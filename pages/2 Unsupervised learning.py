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



tab1, tab2 = st.tabs(["Unsupervised explanation", "Clustering Examples"])





# Interactive graphs of the pre/post data for each marker. 
with tab1:
    st.subheader("INTRO STUFF ")
    st.markdown("Explain unsupervised learning")
        



    st.subheader("PCA and K-means Clustering")

    st.markdown("PCA simplifies complex datasets by reducing the number of features, while keeping as much of the important information so that the significancy of the data is not affected. First, the data is standardised, so that all features are on the same scale. Then key features are identified, through the combination of original features. Finally, the dimensions are reduced, only the top few features are kept which retains the most significant information.")


    st.markdown("PCA is usually followed by a clustering algorithm. K-means clustering is a common way to group data into different categories based on how similar the data points are. It starts with picking the number of groups, with random group centres. The data points are then assigned to the nearest group and the group centres are updates. This is repeated until the best grouping is found.")


    st.markdown("**Example:** Explore PCA and clustering on the Toy dataset below.")

    toy_dat = pd.read_csv('datasets/toy.csv')
    toy_dat['encoded_label'] = [1 if i == 'b' else 0 for i in toy_dat['label'].values]  # Makes a = 0, b = 1.
    del toy_dat['label']    # Drop labels column as not computer readable.

    st.subheader("Raw Toy Dataset:")
    st.dataframe(toy_dat.head())
    st.markdown("The columns, or features, of the toy dataset represent the variables measured for each data point. Whereas, the data points represent individual samples, with each row in the dataset being a different sample. The plot below displays the first two features of the dataset.")

    # Apply PCA
    pca = PCA(n_components=2)
    toydat_pca = pca.fit_transform(toy_dat)
    toy_dat_pca = pd.DataFrame(toydat_pca, columns=["PC1", "PC2"])

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


with tab2:
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


    # Making filename dictionary
    
    cluster_dict = {
    "basic1.csv": {"best_method": ["gmm"], "num_clusters": 4,
                   "description": ""},
    "basic2.csv": {"best_method": ["dbscan"], "num_clusters": 5,
                   "description": "DBSCAN definitely the most effective here (eps=15,min_clusters=5) - the other methods struggle with the elongated shapes of the clusters."},
    "basic3.csv": {"best_method": ["gmm"], "num_clusters": 3},
    "basic4.csv": {"best_method": ["kmeans", "gmm"], "num_clusters": 3},
    "basic5.csv": {"best_method": ["kmeans", "gmm"], "num_clusters": 3},
    "blob.csv": {"best_method": ["kmeans", "gmm"], "num_clusters": 4},
    # "box.csv": {"best_method": "dbscan", "num_clusters": 1}, ###
    # "boxes.csv": {"best_method": "dbscan", "num_clusters": 30}, ###
    "boxes2.csv": {"best_method": ["dbscan"], "num_clusters": 3},
    # "boxes3.csv": {"best_method": "dbscan", "num_clusters": 12}, ###
    "chrome.csv": {"best_method": ["gmm"], "num_clusters": 4},
    "dart.csv": {"best_method": ["gmm"], "num_clusters": 2},
    "dart2.csv": {"best_method": ["dbscan"], "num_clusters": 4},
    "face.csv": {"best_method": ["dbscan"], "num_clusters": 4},
    "hyperplane.csv": {"best_method": ["gmm"], "num_clusters": 2},
    "isolation.csv": {"best_method": ["dbscan"], "num_clusters": 3},
    "lines.csv": {"best_method": ["dbscan"], "num_clusters": 5},
    "lines2.csv": {"best_method": ["dbscan"], "num_clusters": 5},
    "moon_blobs.csv": {"best_method": ["dbscan"], "num_clusters": 4},             ############## check best method
    "network.csv": {"best_method": ["kmeans", "gmm", "dbscan"], "num_clusters": 5},
    "outliers.csv": {"best_method": ["gmm"], "num_clusters": 2},
    # "ring.csv": {"best_method": "kmeans", "num_clusters": 1}, ###
    "sparse.csv": {"best_method": ["kmeans"], "num_clusters": 3},
    "spiral.csv": {"best_method": ["dbscan"], "num_clusters": 1}, ###
    "spiral2.csv": {"best_method": ["gmm"], "num_clusters": 2},
    "spirals.csv": {"best_method": ["gmm"], "num_clusters": 3},
    "supernova.csv": {"best_method": ["gmm"], "num_clusters": 4},
    "triangle.csv": {"best_method": ["gmm"], "num_clusters": 3},
    "un.csv": {"best_method": ["gmm"], "num_clusters": 2},
    "un2.csv": {"best_method": ["dbscan"], "num_clusters": 3},
    "wave.csv": {"best_method": ["gmm"], "num_clusters": 4}
}







    ################################################## 

    def get_data(seed):
        random.seed(seed)
        random_file = random.choice(list(cluster_dict.keys())) 
        # random_file = (list(cluster_dict.keys()))[6]
        # st.write(random_file)

        random_data = pd.read_csv(f"datasets/cluster data/{random_file}")
        random_data = random_data.drop(columns=["color"])
        
        # random_noise = np.random.normal(0, 1)
        # random_data += random_noise
        
        # Add noise??
        
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
    st.write(f"Session seed counter: {session_counter}")


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

    # Option to show data
    st.write("this is a bit irrelevant")
    if st.checkbox('Show dataframe'):
        st.write(random_data.head(5))



    st.write("Make this lil quiz **harder** by increasing the noise of the data.\
             This will make the clusters less distinct. ")
    noise_std = st.slider("Select noise distribution", 0.0, 20.0, step=1.0 )
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
        st.write("running gmm")
        
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=num_centres).fit(data)
        colour_labels = gmm.predict(data)
        
        probabilities = gmm.predict_proba(data)
        size = 15 * probabilities.max(axis=1) ** 2
        
        return size, colour_labels


    def dbscan_cluster(data, eps, min_samples):
        # st.write("running dbscan")
        from sklearn.cluster import DBSCAN
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = dbscan.labels_
        
        num_labels = len(np.unique(labels))
        st.write(f"Number of unique labels: {num_labels}")

        return labels


    # not sure if this works
    def hdbscan_cluster(data):
        st.write("running hdbscan")
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

        clustering_technique_options = ["None", "K-means", "GMM", "DBSCAN", "HBDSCAN"]
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
            st.write("Change the following slider to see how this parameter effects the DBSCAN results.")
            eps = st.slider("Eps", 1, 15)
            min_clusters = st.slider("min_clusters", 1, 15)
            if st.checkbox("Click to see what 'Eps' is....."):
                st.write("Eps is a DBSCAN-SPECIFIC DEFINITION: :nerd_face:")
                st.write("**Eps:** *The maximum distance between two samples for one to be considered as in the neighborhood of the other.*  ")
            
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
            
        basic_plot(random_data, ax1, size, colour_labels, cluster_centres)
        st.pyplot(fig1)
            
        
    # st.write("previous data slider bool:", st.session_state["previous_data_slider"])
        st.write("counter:", st.session_state["counter"])
    

    with col2:
        st.write(f"filename: {random_file}")   
    
        # Quiz.
        st.subheader("Quiz time!")
        question_one = st.radio(
            "Hhow many clusters would you say there are?",(np.arange(1,7)), index=None)
        if question_one == cluster_dict[random_file]["num_clusters"]:
            # get emoji
            st.success("I agree with you :smile:!")
            
            question_two = st.radio(
            "Now, what method is best in determining this?",
            ("kmeans", "gmm", "dbscan", "hbdscan"), index=None)
            
            if question_one == None:
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
        
 


    




    st.subheader("CLUSTER SELECTION")


    end_multi = '''
    From this exercise I hope you now feel like you understand clustering better.  
    There is evidently no one-size-fits-all clustering technique which can be used to cluster any set of data -
    the methods introduced here all have their pros and cons. 
    
    Overall, I would say GMM is the most reliable, in particular for noise-less data. 
    However, it does struggle with clusters that are funky shapes if there is more noise and overlap between clusters. 
    
    It is important to remember that clusters are often hard to distinguish by eye. 
    This exercise uses fake data to allow easy comparison between the accuracy of the different methods in a more interesting way. 
    However, unfortunately, it is rare for a dataset to have the shape of a smiley face in real-life datasets :disappointed: 
    
    To finish off, I'll give a little rundown of the pros of each method:
    
    '''

    st.write(end_multi)

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
        '''
        multi_cons = '''
        - Only good with blobby data - can't do interesting shapes
        '''
        pros_and_cons(multi_pros, multi_cons)
        
    with gmm_tab:
        multi_pros = '''
        - Simple to implement  
        - Good with blobby data 
        - Good with interestingly-shaped clusters if distinct
        - Shows probability of each point being in its designated cluster
        '''
        multi_cons = '''
        - Bad with intertwined data
        '''
        pros_and_cons(multi_pros, multi_cons)
        
    with dbscan_tab:
        multi_pros = ''' 
        - Good with strangely shaped clusters 
        - Shows which points are likely noise
        - Number of clusters does not need to be pre-determined
        '''
        multi_cons = '''
        - Eps must be chosen carefully
        - Can produce an extremely larger number of clusters
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


    
    

    st.write("selecting the correct number of clusters can be done in some way")
    st.write("brief overview")
    st.write("however we wont go into much detail here")
    st.write("here are some links tho")






