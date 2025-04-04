## Neural Network Page 
## Written by Elliot Ayliffe 
## Date: 05/03/24

# Import libraries
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
import io


# Import functions for the Neural Network Page
from Modules.neural_network_functions import (
    nn_architecture_draw,
    activation_function_plot,
    display_image,
    reg_quiz,
    incorrect_reasons,
    dataset_generator,
    dataset_plotter,
    parameter_selector,
    build_nn_model,
    train_nn_model,
    plot_loss,
    decision_boundary_plotter,
    model_evaluation_metrics,
    python_nn_code 
)

# set page layout 
#st.set_page_config(page_title="Neural Networks", layout="wide")

# set title 
st.title("Neural Networks")
# Set tabs for sections within the NN page 
sections = st.tabs(["Tool Information", "Background", "Model Configuration and Training", "Python Implementation"])

# Section 1: Tool Information 
with sections[0]:
    st.header("Build and Explore Your Own Neural Networks for Classification 🧠")

    # Descibe the purpose of the tool 
    with st.expander("**TOOL PURPOSE:**"):
        st.markdown(
            """
        This interactive Neural Network Tool enables users of all skill levels to configure, visualise, and explore
        there own neural network models through an intuitive interface. Navigate the following tabs to learn about various 
        components and parameters of neural networks, and examine how they influence the model's behaviour/outcomes
        and classification performance. This tool offers a hands-on, visual learning experience, introducing users to neural
        networks without the need for time-consuming programming. If you want to learn about the code, you can do that too, 
        under the **Python Implementation** tab.
        """)


    # Describe how to use the tool 
    with st.expander("**USER GUIDE:**"):
        st.markdown("""
        **1. Choose Your Dataset:**     
                        
        Select and adjust your dataset from the provided options, which include computer-generated datasets (Moons, Circles, Blobs).
        All datasets consist of 2 features and a target label for binary and multi-class classification tasks. 
        The computer-generated datasets allow you to customise the number of samples, noise level, and number of centres (for blobs).
        The dataset is then plotted.
        
        **2. Model Configuration:**
                    
        Take time to build and adjust various parameters and hyperparameters of your neural network before you start training. Customisable parameters 
        include:
        - Test dataset size 
        - Number of layers/ neurons 
        - Learning rate 
        - Activation function
        - Regularisation 
        - Optimisation algorithm
        - Epochs 
                    

        Hover over the **'?'** symbol next to each parameter for a hint on what the parameter does.       
        Once you are happy with your settings, click the **BUILD & TRAIN MODEL** button to begin!
                    
        **3. Explore the Model Output and Performance:**
                    
        Analyse your model's performance through a variety of evaluation metrics, loss curves, and execution time, while visualising 
        the decision boundary to gain insights into its behaviour. 

        **4. Repeat:**
                    
        Try out as many different configurations as you wish, hopefully gaining an intuition for the impact of each parameter on model outputs, 
        while demystifying the workings of neural networks.
                    
        **Enjoy!**
                    
        """)

    # List the tools features 
    with st.expander("**TOOL FEATURES:**"):
        st.markdown("""
                    **1. Neural Network Background Information:**

                    Learn about the main components and parameters of neural netorks in the **Background** tab.
                    Engage with interactive dropdowns, visualisations and a quiz to begin to understand the foundations of:

                    - Neural Network Architectures
                    - Activation Functions
                    - Loss an Optimisation Algorithms 
                    - Regularisation Methods
                    - Evaluation Metrics

                    **2. Build & Train Your Own Neural Network**

                    Customise, build, train and visualise your very own Neural Network model without any complicated coding!
                    Navigate to the **Model Configuration and Training** tab to explore the tool containing the following features:

                    - **INTERACTIVE USER INTERFACE**

                    Easily control and configure your model through user-friendly streamlit widgets (sliders, dropdowns, buttons)
                    
                    - **REAL-TIME MODEL TRAINING & EXPLORATION**
                    
                    Train the model in real-time with only a click of a button and let the tool build, train and output the results after
                    only a few seconds (depending on your model).

                    - **PERFORMANCE VISUALISATION & MONITORING**

                    Visualise the model performance through loss curves, a decision boundary plot, and evalution metrics. Use these to help guide possible 
                    improvements for your next model.

                    **3. Learn How to Code Neural Networks in Python**

                    Navigate to the final tab to see how to code a simple neural network classifer using the popular machine learning library, **TensorFlow**.

                    
                    """)

    # Insert an image of a basic architecture 
    st.image("images/3_neural_network_images/nn_architecture.svg")

# Background information section describing the main components of Neural Networks 
with sections[1]:

    st.header("Background information on the main components of Neural Networks 📖")
    st.markdown(""" -----""")

    ## 1. Architectures
    st.subheader("1. Architecture: Layers and Neurons")
    st.markdown("""
    Neural Network architectures determine how neurons and various types of layers are arranged and how information passes between them.
    This structure enables neural networks to identify and model complex patterns within datasets. Common types of architecture include:
    
    - **Feedforward Networks:** Information moves in one direction, for straightforward tasks like classification.
    - **Convolutional Networks:** For capturing spatial patterns in grid-like data like images
    - **Recurrent Networks:** For sequential data like text or time-series (retains memory of previous inputs)
    """)

    # Explain what a neuron is
    with st.expander("**WHAT IS A NEURON?**"):
        st.markdown("""
        Neurons are the basic information-processing units that take inputs, process them, and pass outputs to the neurons in the next layer.
        Individual neurons learn to detect specific features and patterns in data and involve 3 main steps:
        
        **1. Weighted Summation**: Each input is multiplied by a weight that reflects its importance, and then summed.
                    
        **2. Activation Function**: The weighted input sum is passed through an activation function.
                    
        **3. Output:** The activation value is produced and passed as an input to the following layer.
        """)

        # Insert Neuron image 
        st.image("images/3_neural_network_images/Neuron.png")

    # Explain what a layer is 
    with st.expander("**WHAT IS A LAYER?**"):
        st.markdown("""
                    Layers consist of many neurons that perform different stages of feature extraction and processing: The 3 types for a
                        simple architecture are:
                    
                    - **Input Layer:** Receives the raw data features only and then passes them on to the next layer (no processing)
                    - **Hidden Layers:** These layers handle most of the processing by applying weights and activations functions.
                        Having multiple hidden layers enables the network to learn complex, hierarchical representations, with each 
                    layer capturing distinct features of the data.
                    - **Output Layer:** Transforms the processed data into the appropriate format for the specific task (e.g. classification/regression),
                    outputting the final results.
                    """)

    st.write("Input the number of neurons per layer to visualise a simplified architecture.")
    
    # Ask user to input an architecture shape 
    user_architecture_input = st.text_input("Enter a Neural Network architecture (e.g 2,3,2 for 3 layers)")
    
    if user_architecture_input:
        # Split the input (str) to convert into a list of integers (compatibile with the function)
        try:
            architecture_shape = [int(n.strip()) for n in user_architecture_input.split(',')]

            if len(architecture_shape) > 0:
                # call the architecture draw function 
                architecture_diagram = nn_architecture_draw(architecture_shape)
                st.pyplot(architecture_diagram, use_container_width=False)  # plot the figure in streamlit

            else:
                st.write("This is an invalid architecture shape.")
        except ValueError:
            st.write("The input you have provided is invalid. Please follow the correct format (e.g. 2,3,2).")


    ## 2. Activation Functions
    st.markdown(""" -----""")
    st.subheader("2. Activation Functions")
    st.markdown("""
                    An activation function is a mathematical function applied to the output of a neuron, determining how it should
                        be transformed before passing it to the next layer. Their purpose is to add non-linearity to enable neural networks
                    to learn complex patterns and relationships. Without them, only linear relationships could be modelled. 
                    
                    **Explore the common types of activation function below:** """)

    
    # Let user choose an input x value and an activation function for visualisation
    user_input_x = st.slider("Choose an x value as input:", -10, 10, 0)
    user_activation = st.selectbox("Select an Activation Function:", ["Linear", "Sigmoid", "Softmax", "Tanh", "ReLU"])

    # Formatting page 
    column1, column2 = st.columns([1.5,1])

    with column1: 
        # call activation function plotter 
        activation_function_plot(user_activation, user_input_x)

    with column2:

        # Display information (e.g. equation, when to use etc.. for the selected activation function)
        if user_activation == "Linear":
            st.markdown(""" 
                        **WHAT IT DOES:** The linear activation function directly outputs the input value without any non-linear transformations.

                        **EQUATION:** $f(x) = x$

                        **WHEN TO USE:** Typically in the output layer for **Regression** tasks to produce continuous values rather 
                        than distinct classes. It is unbounded, so allows a wide range of predictions.
                        """)
            
        if user_activation == "Sigmoid":
            st.markdown(""" 
                        **WHAT IT DOES:** The sigmoid function squashes the input into a range of 0 to 1 range which is often used to interpret the output as a probability.

                        **EQUATION:** $f(x) = 1/{1+e^{-x}}$

                        **WHEN TO USE:** Typically used in the output layer for **Binary Classification** tasks, where the network outputs
                        a probability of the input being a specific class (e.g. a 0 or 1).
                        """)
            
        if user_activation == "Softmax":
            st.markdown(r""" 
                        **WHAT IT DOES:** The softmax function transforms outputs into a probability distribution across multiple classes outputting 
                        values from 0 to 1 which sum to equal 1.

                        **EQUATION:** $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
                        

                        **WHEN TO USE:** Typically used in the output layer for **Multi-class Classification** tasks, where you need to classify inputs into 
                        multiple categories.
                        """)
        
        if user_activation == "Tanh":
            st.markdown(""" 
                        **WHAT IT DOES:** The Tanh function (Hyperbolic Tangent) squashes the input into a range of -1 to 1 (similar to sigmoid but rescaled). 
                        Centering the data around 0 can give speed advantages when training. 


                        **EQUATION:** $f(x) = e^{x} - e^{-x} / e^{x} + e^{-x}$

                        **WHEN TO USE:** Typically used in hidden layers when data is required to be centered around 0 and for tasks where positive/negative values 
                        need to be captured (e.g. Time series or NLP)
                        """)
        
        if user_activation == "ReLU":
            st.markdown(""" 
                        **WHAT IT DOES:** The ReLU (Rectified Linear Unit) outputs 0 for any negative input (threshold=0) and returns the input 
                        itself for positive numbers (like the linear activation function).

                        **EQUATION:** $f(x) = max(0, x)$

                        **WHEN TO USE:** It is very commonly used in hidden layers as it helps to prevent the vanishing gradient problem, especially 
                        in **deep neural networks**.
                        """)
    

    ## 3. Loss & Optimisation Algorithms 
    st.markdown(""" -----""")
    st.subheader("3. Loss & Optimisation Algorithms")

    # Loss background 
    with st.expander("**WHAT IS LOSS?**"):
        st.markdown("""
                    Loss is a metric that quantifies how well the model is performing by measuring the difference between the model's predictions 
                    and the actual target values. A low loss means predictions are close to the actual targets, with a high loss suggesting
                    the predictions are way off.

                    The network adjusts its weights and biases during training using optimisation algorithms in an attempt to minimise the loss function.

                    We consider two types of loss:

                    - **Training Loss:** Loss calculated from the training dataset during training to help guide model optimisation.
                    - **Validation Loss:** Loss calculated from the (unseen) validation dataset after each epoch to measure how well the model generalises to new data.

                    If the training loss decreases while the validation loss increases, it is a sign that the model is overfitting 
                    (i.e. the model is learning the patterns of the training data too well such that it performs poorly on the validation data).
                    """)
        
        # Insert Loss diagram 
        st.image("images/3_neural_network_images/loss.png")

    # Allow user to explore different types of Loss function
    user_loss = st.selectbox("Select a Loss Function:", ["Mean Squared Error", "Binary Cross-Entropy", "Categorical Cross-Entropy"])

    if user_loss == "Mean Squared Error":
        st.latex(r"MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2")
        st.markdown("""
                    - $N$ = Number of samples in the dataset
                    - $y_i$ = Observed value
                    - $\hat{y_i}$ = Predicted value
                    """)
        st.markdown("""
                    **WHAT IT DOES:** Mean Squared Error (MSE) is a loss function that calculates the average of the squared differences
                    between the predictions and observed values.

                    **WHEN TO USE:** MSE is typically used for **Regression** task as it is effective with continuous data.
                    """)
    
    if user_loss == "Binary Cross-Entropy":
        st.latex(r"Loss = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i})]")
        st.markdown("""
                    - $N$ = Number of samples in the dataset 
                    - $y_i$ = True label (e.g. 1 or 0)
                    - $\hat{y_i}$ = Predicted probability that the $i^{th}$ sample belongs to class 1 (between 0 and 1)
                    """)
        st.markdown("""
                    **WHAT IT DOES:** Binary Cross-Entropy (BCE) is a loss function that measures the difference between the predicted 
                    probabilities and the true labels (e.g. 0 or 1).

                    **WHEN TO USE:** BCE is typically used for **Binary Classification** tasks with only two possible outputs.
                    """)
        
    if user_loss == "Categorical Cross-Entropy":
        st.latex(r"Loss = -\sum_{i=1}^C y_i \log(\hat{y_i})")
        st.markdown("""
                    - $C$ = Total number of classes
                    - $y_i$ = True label (e.g. 1 for the true class, 0 for the other classes)
                    - $\hat{y_i}$ = Predicted probability that the input belongs to class $i$
                    """)
        st.markdown("""
                    **WHAT IT DOES:** Categorical Cross-Entropy (CCE) is a loss function that measures the difference between the predicted
                    probabilities and the true label for each class.

                    **WHEN TO USE:** CCE is typically used for **Multi-class Classification** tasks with several possible outputs.
                    """)
        
    # Optimisation Algorithms background
    with st.expander("**WHAT ARE OPTIMISATION ALGORITHMS?**"):
        st.markdown("""
                    Optimisation algorithms are methods used to minimise the loss function by iteratively adjusting the models parameters. 
                    They rely on **gradient descent**, a technique that computes the gradient of the loss 
                    function with respect to each weight and bias. This gradient then guides the direction in which
                    the parameters should be updated until optimal parameters are found.

                    Popular optimisation algorithms include:

                    - **Stochastic Gradient Descent (SGD)**
                    - **Adam (Adaptive Moment Estimation)**
                    """)
        
        # Insert gradient descent image
        st.image("images/3_neural_network_images/gradient_descent.png")

    
    # Allow user to explore information on two different optimisation algorithms
    user_optimiser = st.selectbox("Select an Optimisation Algorithm:",["Stochastic Gradient Descent", "Adam"])

    if user_optimiser == "Stochastic Gradient Descent":
        st.markdown("""
                        **WHAT IT DOES:** SGD is an optimisation algorithm that updates the parameters using the gradient of the loss function
                        based on a **single randomly selected sample** at each step. This is faster than normal gradient descent which uses the entire 
                        dataset each time.

                        **WHEN TO USE:** Typically used for large datasets as it is faster and more memory efficient.

                        **PARAMETER UPDATE RULE:** 
                        """)
        st.latex(r"\theta = \theta - \alpha \cdot \nabla L(\theta)")
        st.markdown("""
                    - $\\theta$ = Parameters (weights and biases)
                    - $\\alpha$ = Learning rate (step size)
                    - $\\nabla L(\\theta)$ = Loss function gradient
                    """)
    
    if user_optimiser == "Adam":
        st.markdown("""
                    **WHAT IT DOES:** Adaptive Moment Estimation (Adam) is an optimisation algorithm that combines the advantages of two methods:

                    **1. Momentum:** Takes in to account past gradients to help smooth out parameter updates and reduce oscillations.
                    
                    **2. RMSProp:** Adapts the learning rate (step size) for each parameter based on previous gradient magnitudes.

                    **WHEN TOU USE:** Very widely used for many tasks, especially when the optimal learning rate is unknown. It is effective 
                    within deep, complex networks with lots of parameters due to its adaptive learning rate.

                    **PARAMETER UPDATE RULES:**
                    """)
        
        st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2")
        st.latex(r"\theta = \theta - \frac{\alpha {m_t}}{\sqrt{{v_t}} + \epsilon}")

        st.markdown("""
                    - $m_t$ = Exponentially weighted average of gradients (Momentum)
                    - $v_t$ = Exponentially weighted average of squared gradients (RMSPop)
                    - $\\beta_1, \\beta_2$ = Exponential decay rates for averages 
                    - $\\alpha$ = Learning rate (step size)
                    - $\\epsilon$ = constant (stops division by 0)
                    """)

    ## 4. Regularisation 
    st.markdown(""" -----""")
    st.subheader("4. Regularisation")

    # Regularisation background
    st.markdown("""
                Regularistion is a machine learning technique used to prevent **overfitting** by penalising the model's complexity or large weights.
                When a model is overly complex, it can suffer from **memorising the training data** rather than learning patterns and, therefore, perform poorly 
                on unseen data. Regularisation helps the model to **generalise** better to new data.
                """)
    with st.expander("**WHAT IS THE DIFFERENCE BETWEEN OVERFITTING AND UNDERFITTING?**"):
        st.markdown("""
                    - **Overfitting:** When a model learns the training data too well including noise and unnecessary details, rather than 
                    learning overall important patterns.
                        - **Signs:** High accuracy on the training dataset, with low accuracy on the test/validation dataset.
                        - **Potential Fixes:** Add regularisation, simplify the model, early stopping during training, add more training data to encourage better generalisation
                    \n
                    - **Underfitting:**  When the model is overly simple and cannot capture any patterns in the data, resulting in poor performance on all datasets 
                        - **Signs:** Low accuracy on both training and test data, high bias (model makes strong assumptions)
                        - **Potential Fixes:** Increase model complexity, reduce regularisation, train for longer (more epochs)
                    """)
        
    st.markdown("""
                **Common Types of Regularisation:**

                - **L1 (Lasso)**
                    - Adds a penalty equal to the **absolute value** of the weights
                    - Effectively removes less important features by setting certain weights to 0, encouraging sparsity.
                    - Typically used on tasks where only a select few features are important.

                - **L2 (Ridge)**
                    - Adds a penalty equal to the **square value** of the weights 
                    - Stabilises the model by spreading large weights evenly across features
                    - Typically used for tasks that require all features but need model simplification.
                
                - **Dropout**
                    - Randomly deactivates (drops) neurons during each training step, forcing the network to learn many independent paths
                for the same conclusion. 
                    - This makes the model more robust and improves generalisation.
                    - Popular technique used in neural networks
                """)
    
    # Overfitting/ Underfitting Quiz
    st.markdown("#### Overfitting and Underfitting Quiz:")
    st.markdown("""
                Take this short quiz on overfitting and underfitting for classification and regression tasks.
                Look at each graph and decide whether it is an example of overfitting, underfitting, or whether its the right fit. 
                \n**Note:** When selecting "Next Question", press this twice 
                and don't submit a new answer until a new graph has been displayed.
                """)
    
    # Run quiz 
    reg_quiz()


    ## 5. Model Evaluation Metrics 
    st.markdown(""" -----""")
    st.subheader("5. Model Evaluation Metrics")
    
    st.markdown("""
                Various evaluation metrics are used to **assess how well a neural network performs** on the test data (unseen).
                They are used as a guide to see if the model is doing well or making mistakes and therefore needs improvement.
                Classification tasks have various metrics that measure slightly different things. These are shown below.
                """)
    # Define the metrics details and store them in a dictionary
    metrics = {"Metric": ["Accuracy", "Precision", "Recall (sensitivity)", "F1 Score"],
                "Description": ["Percentage of correct predictions",
                                "Proportion of true positives among predicted positives",
                                "Proportion of true positives among actual positives",
                                "Harmonic mean of precision and recall"],
                "Equation": ["$$\\frac{\\text{Correct Predictions}}{\\text{Total Predictions}}$$",
                            "$$\\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$$",
                            "$$\\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$",
                            "$$2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$"],
                "When it's useful": ["When classes are balanced",
                                        "Tasks with many false positives",
                                        "When missing positive cases (false negatives) is costly",
                                        "When you need a balance of precision and recall"]}

    # Create dataframe and convert the table to markdown for display 
    metrics_df = pd.DataFrame(metrics)
    st.markdown(metrics_df.to_markdown(index=False), unsafe_allow_html=True)



# Section 3: Model Configuration and Training
with sections[2]:
    column1, column2 = st.columns(2)    # create a 2-column interface 
    with column2:
        st.subheader("Model Outputs:")

    with column1:
        st.subheader("Model Configuration:")

        # Dropdown for user to select a dataset 
        data = st.selectbox("Select a Dataset", ["Circles", "Moons", "Blobs"])

        # Reveal more dataset parameters for circles, moons and blobs data (for user input)
        if data in ["Circles", "Moons"]:
            sample_num = st.slider("Select the Number of Samples", min_value=20, max_value=1000, value=300, step=20)
            noise = st.slider("Select the Level of Noise", min_value=0.0, max_value=0.4, value=0.08, step=0.01)

            # Generate the synthetic data using specified parameters 
            X, y = dataset_generator(data, sample_num, noise)

        elif data == "Blobs":
            sample_num = st.slider("Select the Number of Samples", min_value=20, max_value=1000, value=300, step=20)
            noise = st.slider("Select the Level of Noise", min_value=0.0, max_value=0.4, value=0.08, step=0.01)
            n_centres = st.number_input("Select the Number of Centres",2,4)

            # Generate the synthetic data using specified parameters 
            X, y = dataset_generator(data, sample_num, noise, n_centres=n_centres)


        # Plot the selected dataset 
        dataset_plotter(X,y)


        # Call parameter selector function to store user selected parameters 
        test_size, learning_rate, activation_function, n_hidden_layers, n_neurons_per_layer, regularistion_method, reg_rate, optimiser, n_epochs = parameter_selector()


        # Button to build and train the model
        if st.button("**BUILD & TRAIN MODEL**", type="primary", use_container_width=True):
            

            # Build the neural network using the user's selected parameters (call function)
            model = build_nn_model(X, y, test_size, learning_rate, activation_function, n_hidden_layers, n_neurons_per_layer, regularistion_method, reg_rate, optimiser)
            
            # Store the nn model in session state to access it for training
            st.session_state['model'] = model

            # Display model summary 
            st.markdown("**Model Summary:**")
            io_file = io.StringIO()
            model.summary(print_fn=lambda x: io_file.write(x + '\n'))
            summary_text = io_file.getvalue()
            formatted_summary = f"""
            <div style="max-width: 320px; overflow-x: auto;">
                <pre>{summary_text}</pre>
            </div>
            """
            st.markdown(formatted_summary, unsafe_allow_html=True)


            # Split into training and testing data and call the training function
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model_trained, history = train_nn_model(model, X_train, y_train, X_test, y_test, n_epochs)

            st.markdown("*Have a look at your model outputs in the right-hand column.*")

            with column2:
                    # Plot loss curves
                    plot_loss(history)

                    # Plot decision boundary 
                    st.markdown("#### Decision Boundary:")
                    st.markdown("This may take a few seconds to load.")
                    decision_boundary_plotter(model_trained, X_train, y_train, X_test, y_test)


                    # Compute and display model evaluation metrics
                    st.markdown("#### Performance Evaluation:")
                    model_evaluation_metrics(model_trained, X_train, y_train, X_test, y_test)
            
    with column2:
        st.markdown(""" -----""")
        st.image("images/3_neural_network_images/brain.png")




# Section 4: Python Implementation 
with sections[3]:
    python_nn_code()
    


        









