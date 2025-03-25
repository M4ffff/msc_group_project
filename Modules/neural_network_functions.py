## Interactive Machine Learning Platform 
## Neural Networks Page Module 
## Written by Elliot Ayliffe 
## Date: 05/03/2025

## Module containing functions to run the Neural Network page within the Streamlit app

# Import libraries (TensorFlow must be installed)
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


# Imports for the 'Background' tab
import random 
import os 
from PIL import Image

# Imports for 'Model Configuration and Training' tab 
from sklearn.datasets import make_moons, make_circles, make_blobs 
from sklearn.model_selection import train_test_split
# import tensorflow as tf 
# from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Input
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.callbacks import Callback 
import io
from time import time
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score






# ------------------------------------------------------------------------------------------------------------------------------

#### FUNCTIONS FOR TAB 2: 'Background' ####

# ------------------------------------------------------------------------------------------------------------------------------


# Function to draw a simplified neural network architecture based on the users inputs 
def nn_architecture_draw(architecture_shape):
    """
    Creates a visualisation of a simplified neural network architecture (layers and neurons) based 
    on the users inputs.

    Args:
        architecture_shape (list of int): User inputted list of integers that represent the number of neurons 
                                          in each layer (e.g. 2,3,1) will generate an architecture of 3 layers 
                                          containing 2 neurons in the input layer, 3 neurons in the hidden layer,
                                          and 1 neuron in the output layer.
    
    Returns:
        A figure of a simplified neural network architecture to facilitate understanding                                
    """

    # Create the figure (adjusts its size based on the size of input)
    fig, ax = plt.subplots(figsize=(len(architecture_shape)*2, len(architecture_shape)*0.5), dpi=200)

    # Define the spacings between layers and neurons for formatting purposes 
    layer_space = 0.8
    neuron_space = 0.3

    # Loop over each layer 
    for layer, neurons in enumerate(architecture_shape):

        # Determine the y posistion of the current layer 
        layer_y_pos = layer * layer_space 

        # Determine the x positions for the neurons within the current layer 
        neurons_x_pos = [(n - (neurons - 1)/2) * neuron_space for n in range(neurons)]

        # Draw each neuron based on x and y coordinates (circles)
        for x_pos in neurons_x_pos:
            ax.add_patch(plt.Circle((x_pos, layer_y_pos), color='green', fill=True, radius=0.08))

        # Label each layer 
        label_pos = 0.2     # Formatting for text 

        if layer == 0:  # Input layer 
            ax.text(max(neurons_x_pos) + label_pos, layer_y_pos, 'Input layer', fontsize=6, verticalalignment='center', horizontalalignment='left')

        elif layer == len(architecture_shape)-1: # Output layer 
            ax.text(max(neurons_x_pos) + label_pos, layer_y_pos, 'Output layer', fontsize=6, verticalalignment='center', horizontalalignment='left')

        else:   # Hidden layers (all middle layers)
            ax.text(max(neurons_x_pos) + label_pos, layer_y_pos, f'Hidden layer {layer}', fontsize=6, verticalalignment='center', horizontalalignment='left')

        # Draw the lines connecting each neuron from layer to layer 
        if layer < len(architecture_shape) - 1:
            following_layer_y_pos = (layer + 1) * layer_space   # Determine y posisition of the following layer 
            # Determine the x position for each neuron in the following layer 
            following_neurons_x_pos = [(n - (architecture_shape[layer + 1] - 1) / 2) * neuron_space for n in range(architecture_shape[layer + 1])]

            # Draw lines connecting the neurons in the current layer to the neurons in the following layer 
            for x_pos in neurons_x_pos:
                for following_x_pos in following_neurons_x_pos:
                    ax.plot([x_pos, following_x_pos], [layer_y_pos, following_layer_y_pos], color='black', linewidth=0.4)

    # Format figure 
    ax.axis('off')
    ax.set_aspect('equal')

    return fig 


# Function to plot user-selected activation functions for visualisation
def activation_function_plot(type, x_input):
    """
    Plots the activation function specified by the user and takes in an input value x
    to compute (and visualise) the output.

    Args:
        type (str): Type of activation function the users want to explore (e.g. ReLU, Sigmoid etc..)
        x_input: Input value x to compute the output value y.

    Returns: Plot of the activation function showing the output of an inputted x-value.
    """
    # Define x-values 
    x = np.linspace(-10, 10, 100)

    if type== "Sigmoid":
        f_x = 1 / (1 + np.exp(-x))  # all y values
        y_output = 1 / (1 + np.exp(-x_input))   # singular y value computed from x-input 

    if type== "ReLU":
        f_x = np.maximum(0,x)
        y_output = np.maximum(0,x_input)

    if type=="Softmax":
        f_x = np.exp(x) / np.sum(np.exp(x))
        y_output = np.exp(x_input) / np.sum(np.exp(x))

    if type=="Tanh":
        f_x = np.tanh(x)
        y_output = np.tanh(x_input)

    if type=="Linear":
        f_x = x
        y_output = x_input

    # Plot activation function and user input x
    fig, ax = plt.subplots()
    ax.plot(x,f_x)  # plot function 
    ax.axvline(x_input, linestyle='--', linewidth=0.5, color='red', label=f"Input: {x_input}")  # plot input line 
    ax.axhline(y_output, linestyle='--', linewidth=0.5, color='green', label=f"Output: {y_output:.3f}")  # plot output line 
    ax.set_title(f"{type}")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend()
    st.pyplot(fig)
    


### Functions for Regularisation section quiz 
def display_image(img_filename):
    """
    Loads the image corresponding to the current quiz question (overfitting/underfitting
    quiz in the regularisation section)
    """
    path = os.path.join("images/reg_quiz_images", img_filename)
    image = Image.open(path)        # locate and open the image 
    st.image(image)         # insert the image 



# Define the images to be used in the quiz 
quiz_images = ['overfitting_classification.png', 'underfitting_classification.png', 
                       'overfitting_regression.png', 'underfitting_regression.png', 
                       'right_fit_classification.png', 'right_fit_regression.png']

# list the correct answers for the quiz which are mapped the image filenames
correct_answers = {
            'overfitting_classification.png': '**Overfitting (classification)**',
            'underfitting_classification.png': '**Underfitting (classification)**',
            'overfitting_regression.png': '**Overfitting (regression)**',
            'underfitting_regression.png': '**Underfitting (regression)**',
            'right_fit_classification.png': '**Right fit (classification)**',
            'right_fit_regression.png': '**Right fit (regression)**'}

def reg_quiz():
    """
    This function creates and organises the quiz layout and navigation:
    - displays question (image)
    - user options (answer submission, next question)
    """

    # Initialise the quiz session state 
    if 'started' not in st.session_state:   # if quiz hasn't started yet set the initial variables 
        st.session_state.started = False
        st.session_state.score = 0  # quiz score 
        st.session_state.n_question = 0 # starts at the first question
        st.session_state.user_answer = None    # the user selected answer is reset 
        st.session_state.submitted_answer = False 

    # Show button to start the quiz (if it hasn't started yet)
    if not st.session_state.started:
        if st.button("BEGIN QUIZ", type="primary"):
            
            # Update the session state, once the button is pressed
            st.session_state.started = True 
            st.session_state.score = 0
            st.session_state.n_question = 0
            st.session_state.user_answer = None
            st.session_state.submitted_answer = False

    # show the image and answers if the quiz has started 
    if st.session_state.started:
        question_image = quiz_images[st.session_state.n_question]  # retrieve the image using the question number

        # display image 
        display_image(question_image)

        # Possible answers for the user
        answer_options = ["**Overfitting (classification)**",
                          "**Right fit (classification)**",
                          "**Underfitting (classification)**",
                          "**Overfitting (regression)**",
                          "**Right fit (regression)**",
                          "**Underfitting (regression)**"]
        
        # Allow user to choose answer (use key to ensure each radio button is unique for each question)
        selected_answer = st.radio("Look at the graph and select the correct answer:", 
                                   answer_options, 
                                   key=f"radio_{st.session_state.n_question}")
        
        # update session state with chosen answer 
        st.session_state.user_answer = selected_answer 

        # allow user to click a "submit answer" button
        submit_button = st.button("Submit Answer")
        if submit_button and st.session_state.user_answer:
            st.session_state.submitted_answer = True

            # Determine whether the answer is correct or not 
            if st.session_state.user_answer == correct_answers[question_image]:
                st.session_state.score += 1     # if correct, add 1 point to the score 
                st.success("Correct! Well done.")
            else:       # if incorrect give the correct answer with reasoning 
                st.error(f"Incorrect. The correct answer is: {correct_answers[question_image]}.\n{incorrect_reasons(question_image)}")

        # After the answer is submitted and marked, display the next question button
        if st.session_state.submitted_answer:
            next_q_button = st.button("Next Question")
            if next_q_button:
                st.write("Please click the 'Next Question' button again to display a new graph.")
                if st.session_state.n_question < len(quiz_images) - 1:          # check if the question is the last one or not
                    # update/reset session sate 
                    st.session_state.n_question += 1
                    st.session_state.user_answer = None
                    st.session_state.submitted_answer = False 

                else:       # if its the last question, end the quiz
                    st.session_state.started = False 
                    # show user their score 
                    st.markdown(f"#### Score: {st.session_state.score}/{len(quiz_images)}\nGood Effort!")
                    # give button to try the quiz again 
                    if st.button("Try Again"):
                        st.session_state.score = 0
                        st.session_state.n_question = 0
                        st.session_state.started = True

def incorrect_reasons(img_filename):
    """
    This function stores the explanations given to users if they get an answer wrong.
    It returns the corresponding explanation to the specific image passed.
    """
    explanations = {
        'overfitting_classification.png': 'This is overfitting as the classification model (in black) fits the data too closely.',
        'underfitting_classification.png': 'This is underfitting as the classification model is too simple to capture the underlying patterns.',
        'overfitting_regression.png': 'This is overfitting as the regression model fits the data too closely, capturing unwanted noise.',
        'underfitting_regression.png': 'This is underfitting as the regression model is too simple to capture the overall trend.',
        'right_fit_classification.png': 'A right fit model correctly captures the relationship without overfitting or underfitting.',
        'right_fit_regression.png': 'A right fit regression model correctly captures the data’s trend without overfitting or underfitting.'
    }
    return explanations.get(img_filename)




# ------------------------------------------------------------------------------------------------------------------------------

#### FUNCTIONS FOR TAB 3: 'Model Configuration and Training' ####

# ------------------------------------------------------------------------------------------------------------------------------


# Function to generate circles, moons and blobs datasets
def dataset_generator(user_selected_dataset, sample_num, noise, n_centres=2):
    """
    Function that generates synthetic datasets (circles, moons, blobs) using user-selected parameters.
    The generated dataset is 2D containing 2 features, X, and a binary label, Y, (0 or 1)
    
    Args:
        user_selected _dataset (str): The dataset type specified by the user ("Circles", "Moons", "Blobs"). This will be generated.
        sample_num (int): The number of samples to be included in the dataset (number of points generated)
        noise (float): The level of noise to apply to the data.

    Returns:
        X (numpy.ndarray): The Feature matrix (x1 and x2). Input data
        y (numpy.ndarray): Labels (binary classification, 0 or 1)
    """
    if user_selected_dataset == "Circles":
        X, y = make_circles(n_samples=sample_num, noise=noise, random_state=42) # generates a small circle inside a larger circle 
    
    elif user_selected_dataset == "Moons":
        X, y = make_moons(n_samples=sample_num, noise=noise, random_state=42) # generates two interleaving half circles 

    elif user_selected_dataset == "Blobs":
        X, y = make_blobs(n_samples=sample_num, n_features=2, centers=n_centres, cluster_std=noise*20,random_state=42) # generates n_centres gaussian clusters of points

    return X, y


# Function to plot the user-selected dataset
def dataset_plotter(X, y):
    """
    Function that plots a dataset as a scatterplot (Feature 1 vs Feature 2, colours = labels )

    Args: 
        X (numpy.ndarray): The Feature Matrix (x1 and x2). Input data, must be shape 2 features
        y (numpy.ndarray): Labels (binary classification, 0 or 1)

    Returns:
        Scatter plot of the dataset
    """
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    st.pyplot(fig)



# Function to collect user input for the model's parameters/ hyperparameters 
def parameter_selector():
    """
    Displays an UI that allows the user to configure the parameters/hyperparameters of their neural network.

    Args: 
        None

    Returns:
        - test_size (float): Proportion of the data to be used as the test data (0.1 to 0.5)
        - learning_rate (float): Learning rate for the training process (0 to 0.1)
        - activation_function (str): The activation function used in the hidden layers 
        - n_hidden_layers (int): The number of hidden layers to use in the neural network architecture (1 to 5)
        - n_neurons_per_layer (int): The number of neurons per hidden layer (10 to 200)
        - regularistion (str or None): The method of regularisation or none
        - reg_rate (float): The regularisation rate (0 to 0.1)
        - optimiser (str): Optimisation algorithm to be used for training 
        - epochs (int): Number of epochs (iterations) for training (10 to 500)
    """

    st.markdown("""
                #### Select the parameters and hyperparameters to build your own Neural Network Model:
                """)
    # Get input from the user
    test_size = st.slider("Test Dataset Size", min_value=0.1, max_value=0.5, value=0.2, step=0.01, help="This is the proportion of the dataset to use for testing (10-50%)")
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, help="This is basically the step size when adjusting parameters during training", format="%.3f")
    activation_function = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"], index=0, help="This will be applied to the hidden layers")
    n_hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=5, value=2, step=1)
    n_neurons_per_layer = st.slider("Neurons per Hidden Layer", min_value=10, max_value=200, value=64, step=10)
    regularisation = st.selectbox("Regularisation Method", ["None", "L1", "L2"], index=0, help="This helps to prevent overfitting")
    reg_rate = st.slider("Regularisation Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, help="How much penalty to give to the complexity of the model", format="%.4f")
    optimiser = st.selectbox("Optimiser", ["SGD", "Adam"], index=1, help="This is the optimisation algorithm to be used for training")
    epochs = st.slider("Epochs", min_value=10, max_value=500, value=100, step=10, help="The number of iterations during training (length of training)")

    return test_size, learning_rate, activation_function, n_hidden_layers, n_neurons_per_layer, regularisation, reg_rate, optimiser, epochs



# Function to build the model
def build_nn_model(X, y, test_size, learning_rate, activation_function, n_hidden_layers, n_neurons_per_layer, regularisation, reg_rate, optimiser):
    """
    Takes the users selected parameters and builds the neural network model. 
    Automatically determines the input dimensions and number of classes from the X,y dataset.

    Args:
        X (numpy.ndarray): The Feature matrix (x1 and x2). Used to determine the input dimensions
        y (numpy.ndarray): Labels. Used to determine the number of classes (required for the output layer)
        test_size (float): Proportion of the data to be used as the test data (0.1 to 0.5)
        learning_rate (float): Learning rate for the training process (0 to 0.1)
        activation_function (str): The activation function used in the hidden layers 
        n_hidden_layers (int): The number of hidden layers to use in the neural network architecture (1 to 5)
        n_neurons_per_layer (int): The number of neurons per hidden layer (10 to 200)
        regularistion (str or None): The method of regularisation or none
        reg_rate (float): The regularisation rate (0 to 0.1)
        optimiser (str): Optimisation algorithm to be used for training 

    Returns:
        nn_model (Sequential): the compiled Sequential model built using Tensorflow 
    """

    # Work out how many classes in the dataset 
    n_classes = len(np.unique(y))

    nn_model = Sequential() 

    # Set the regularisation method (L1, L2, or None)
    if regularisation == "L1":
        reg_method = l1(reg_rate)
    elif regularisation == "L2":
        reg_method = l2(reg_rate)
    else:
        reg_method = None

    # Input layer (convert acitvation function string to lower case)
    # Determine input shape from the number of features in X
    nn_model.add(Input(shape=(X.shape[1],)))
    nn_model.add(Dense(n_neurons_per_layer, activation=activation_function.lower(), kernel_regularizer=reg_method)) #First hidden layer

    # Hidden layers 
    for _ in range(n_hidden_layers-1):
        nn_model.add(Dense(n_neurons_per_layer, activation=activation_function.lower(), kernel_regularizer=reg_method))

    # Output layer (provide sigmoid activation for 2 classes and softmax for greater than 2 classes)
    if n_classes == 2:
        nn_model.add(Dense(1, activation='sigmoid'))

    else:
        nn_model.add(Dense(n_classes, activation='softmax'))

    # Set the optimisation algorithm 
    if optimiser == "Adam":
        opt_algorithm = Adam(learning_rate=learning_rate)
    else:
        opt_algorithm = SGD(learning_rate=learning_rate)

    # Compile model (set the optimisation algorithm and loss function)
    # use binary cross-entropy for 2 classes and categorical cross-entropy for more than 2 classes 
    if n_classes == 2:
        nn_model.compile(optimizer=opt_algorithm, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        nn_model.compile(optimizer=opt_algorithm, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Give message to user
    st.success("**Your Neural Network Model has been built!**")
    return nn_model


# Function to Train the neural network model
def train_nn_model(nn_model, X_train, y_train, X_test, y_test, epochs):
    """
    Trains the neural network on the generated dataset.
    - Times the training process displaying a loading bar for the user.

    Args:
        nn_model (Sequential): the compiled Sequential model built previously to be trained
        X_train (numpy.ndarray): The Feature matrix (x1 and x2) for the training dataset
        y_train (numpy.ndarray): Labels for the training dataset 
        X_test (numpy.ndarray): The Feature matrix (x1 and x2) for the testing dataset
        y_test (numpy.ndarray): Labels for the testing dataset 
        epochs (int): Number of epochs (iterations) for training (10 to 500)

    Returns:
        nn_model (Sequential): The trained model 
        history: training history 
    """
    # Text for the User
    st.balloons()
    st.info("**Training has Begun!**")


    # Use to_categorical to convert integer labels to one-hot encoded if it is multi-class classification (n_classes > 2)
    if len(np.unique(y_train)) > 2 :
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # Initialise a placeholder for the training process total runtime 
    training_total_time = st.empty()
    # Initialise a loading bar so the user can track the progress of training.
    loading_bar = st.progress(0)
    loading_text = st.empty()

    # Create a callback class using Keras to update the loading bar after each epoch has completed 
    class TrainingProgress(Callback):
        def __init__(self, n_epochs):
            # Call the constructor of the parent Callback class
            super().__init__()
            # Store the total number of epochs 
            self.n_epochs = n_epochs

        # Method that is called after each epoch 
        def on_epoch_end(self, epoch, logs=None):
            # calculate the training progress and update the loading bar 
            training_progress = (epoch + 1) / self.n_epochs     # Percentage completed 
            loading_bar.progress(training_progress)
            loading_text.markdown(f"Training is **{int(training_progress * 100)}%** completed")

    # Start timer for training 
    start = time()

    # Train model (evaluate model performance on test set after each epoch)
    # Include the callback for the loading bar 
    history = nn_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[TrainingProgress(epochs)], verbose=0)

    # end timer and calculate the total training time
    total_time =  time() - start
    training_total_time.markdown(f"Total Training Time: **{total_time:.3f} seconds**")

    return nn_model, history 


# Function to plot the training loss and validation against epochs
def plot_loss(history):

    # Retrieve training loss and validation loss from the trained model history
    train_loss = history.history['loss']
    validation_loss = history.history.get('val_loss')

    fig, ax = plt.subplots(figsize=(10,6))

    # plot training loss
    ax.plot(train_loss, color='blue', linewidth=2, label="Training Loss")
    # plot validation loss 
    ax.plot(validation_loss, color='red', linewidth=2, label="Validation Loss")

    # Format plot 
    ax.set_title("Loss vs Epochs", fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True)

    st.pyplot(fig)



# Function to plot decision boundary 
def decision_boundary_plotter(trained_model, X_train, y_train, X_test, y_test):
    """
    Plots the decision boundary of the model with the training and testing data points specified

    Args:
        trained_model (Sequential): Trained model. used to make predictions
        X_train (numpy.ndarray): The Feature matrix (x1 and x2) for the training dataset
        y_train (numpy.ndarray): Labels for the training dataset 
        X_test (numpy.ndarray): The Feature matrix (x1 and x2) for the testing dataset
        y_test (numpy.ndarray): Labels for the testing dataset

    Returns:
        Displays the plot in the streamlit app
    """

    # set grid resolution (set at this to reduce computation time)
    res = 0.05
    # calculate the range for x and y axes
    min_x = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5
    max_x = max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
    min_y = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5
    max_y = max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5

    # Create meshgrid of entire space 
    xx, yy = np.meshgrid(np.arange(min_x, max_x, res), np.arange(min_y, max_y, res))

    # Predict the class for each grid point
    preds = trained_model.predict(np.c_[xx.ravel(), yy.ravel()], batch_size=256, verbose=0)

    # Adjust the predictions based on the number of classes 
    if preds.shape[1] > 1:          # for multi-class 
        preds = np.argmax(preds, axis=1)        # extract the class with the highest probability 
    
    else:       # for binary classification
        preds = (preds > 0.5).astype(int)

    preds = preds.reshape(xx.shape)

    # Plot the decision boundary 
    fig, ax = plt.subplots(figsize=(8,6))
    ax.contourf(xx, yy, preds, cmap='viridis', alpha=0.5)

    # Plot training data and testing data
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', edgecolors='black', label='Training Data')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='^', edgecolors='black', label='Test Data')

    # Format plot 
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.legend(loc='upper right', fontsize=12)

    st.pyplot(fig)




# Function to compute and display the evaluation metrics for the training data and testing data
def model_evaluation_metrics(trained_model, X_train, y_train, X_test, y_test):
    """
    Using the trained model, this function makes predictions for both the training dataset 
    (X_train) and testing dataset (X_test) then computes and displays various performance 
    metrics for both datasets (accuracy, precision, recall, f1 score).

    Args:
        trained_model (Sequential): Trained model. used to make predictions
        X_train (numpy.ndarray): The Feature matrix (x1 and x2) for the training dataset
        y_train (numpy.ndarray): Labels for the training dataset 
        X_test (numpy.ndarray): The Feature matrix (x1 and x2) for the testing dataset
        y_test (numpy.ndarray): Labels for the testing dataset
    """

    # Make predictions on the training dataset
    y_preds_train = trained_model.predict(X_train, verbose=0)
    # Adjust the predictions based on the number of classes
    if y_preds_train.shape[1] > 1:       # multi-class 
        y_preds_train = np.argmax(y_preds_train, axis=1)
    else:
        y_preds_train = (y_preds_train > 0.5).astype(int)

    # Make predictions on the testing data
    y_preds_test = trained_model.predict(X_test, verbose=0)
    # Adjust the predictions based on the number of classes
    if y_preds_test.shape[1] > 1:       # multi-class 
        y_preds_test = np.argmax(y_preds_test, axis=1)
    else:
        y_preds_test = (y_preds_test > 0.5).astype(int)

    # Compute the evaluation metrics using the training data 
    training_accuracy = accuracy_score(y_train, y_preds_train)
    training_precision = precision_score(y_train, y_preds_train, average='weighted', zero_division=1)     # calculate the average of the metric across multiple classes   
    training_recall = recall_score(y_train, y_preds_train, average='weighted', zero_division=1)
    training_f1_score = f1_score(y_train, y_preds_train, average='weighted', zero_division=1)

    # Compute the evaluation metrics using the testing data 
    testing_accuracy = accuracy_score(y_test, y_preds_test)
    testing_precision = precision_score(y_test, y_preds_test, average='weighted', zero_division=1)     # calculate the average of the metric across multiple classes   
    testing_recall = recall_score(y_test, y_preds_test, average='weighted', zero_division=1)
    testing_f1_score = f1_score(y_test, y_preds_test, average='weighted', zero_division=1)

    # formatting into columns 
    metrics_col1, metrics_col2 = st.columns(2)

    # Display the metrics with a visually friendly bar 
    with metrics_col1:
        st.info("**Training Data:**")
        st.markdown(f"**Accuracy:** {training_accuracy * 100:.2f}%")
        st.progress(training_accuracy)
        st.markdown(f"**Precision:** {training_precision:.3f}")
        st.progress(training_precision)
        st.markdown(f"**Recall:** {training_recall:.3f}")
        st.progress(training_recall)
        st.markdown(f"**F1 Score:** {training_f1_score:.3f}")
        st.progress(training_f1_score)
        
    with metrics_col2:
        st.info("**Testing Data:**")
        st.markdown(f"**Accuracy:** {testing_accuracy * 100:.2f}%")
        st.progress(testing_accuracy)
        st.markdown(f"**Precision:** {testing_precision:.3f}")
        st.progress(testing_precision)
        st.markdown(f"**Recall:** {testing_recall:.3f}")
        st.progress(testing_recall)
        st.markdown(f"**F1 Score:** {testing_f1_score:.3f}")
        st.progress(testing_f1_score)




# ------------------------------------------------------------------------------------------------------------------------------

#### FUNCTIONS FOR TAB 4: 'Python Implementation' ####

# ------------------------------------------------------------------------------------------------------------------------------


# Function to display a simplified implementation of a basic Neural Network using Tensorflow 
def python_nn_code():
    """
    Displays a simplified code of how to build, train, predict and evaluate neural networks using 
    Tensorflow in Python. Helps the user to understand the code behind the model if they wish. 
    """

    st.header("How to Code a Neural Network Classifier using TensorFlow 🖋️")
   

    st.markdown("""
                The following code demonstrates how to implement a simple neural network using TensorFlow, 
                guiding you through the 5 key stages:

                - **Preprocessing**
                - **Building**
                - **Training**
                - **Prediction**
                - **Evaluating**
                """)

    st.markdown(""" -----""")

    # preprocessing information 
    st.subheader("Step 1: Loading & Preprocessing ⚙️")
    st.markdown("""
                1. Import the required libraries and load the dataset
                2. Split the data into training and testing sets 
                3. Standardise the features (for better performance)

                ``` python
                import tensorflow as tf 
                from sklearn.model_selection import train_test_split 
                import numpy as np
                import pandas as pd

                # Load data
                data = pd.read_csv('example_dataset.csv')
                X = data.drop('y_label', axis=1)    # Extract the feature matrix
                y = data['y_label']                 # Extract the labels

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Standardise features 
                scale = StandardScaler()
                X_train = scale.fit_transform(X_train)
                X_test = scale.transform(X_test)
                ```
                """)

    # Building information
    st.markdown(""" -----""")
    st.subheader("Step 2: Building the Model 🏗️")
    st.markdown("""
                1. Create the model architecture: Layers and number of neurons (input, hidden, output layers)
                2. Select appropraite activation functions: *ReLU* is a good choice for hidden layers, output layers should use either 
                *sigmoid* (binary classification) or *softmax* (mulit-class classification)
                3. Select the appropriate Regularizer (if any)
                4. Compile the model with your chosen optimisation algorithm (*Adam* or *SGD*), learning rate and loss function:
                *binary_crossentropy* (binary classification) or *categorical_crossentropy* (multi-class).

                ```python
                # Set the neural network architecture
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=X.shape[1],))                                   # Set the input layer 
                    tf.keras.layers.Dense(25, activation='relu', kernal_regularizer=l2(0.001))  # Hidden layer 1 
                    tf.keras.layers.Dense(10, activation='relu', kernal_regularizer=l2(0.001))  # Hidden layer 2
                    tf.keras.layers.Dense(1, activation='sigmoid')                              # Output layer for 2 classes 
                    # tf.keras.layers.Dense(n_classes, activation='softmax')                    # Output layer for > 2 classes
                ])

                # Compile model
                # Binary classification 
                model.compile(optimizer=keras.optimizers.Adam(learning rate), loss='binary_crossentropy', metrics=['accuracy']) 
                # Multi-class classification
                model.compile(optimizer=keras.optimizers.Adam(learning rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                """)
    
    # Training information
    st.markdown(""" -----""")
    st.subheader("Step 3: Training the Model 🏋️‍♂️")
    st.markdown("""
                1. Fit the model on the training dataset
                2. Set the number of epochs (iterations) to train for.
                3. Track performance using the unseen test data (optional)

                ```python
                model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
                ```
                """)

    # Prediction information
    st.markdown(""" -----""")
    st.subheader("Step 4: Making Predictions for the Testing Data 🔮")
    st.markdown("""
                1. Make predictions on the unseen testing data using the newly trained model
                2. Make sure to convert the probabilities to classes for both binary and mult-class tasks

                ```python
                # Predict y labels for the test data 
                y_test_predictions = model.predict(X_test)

                # Convert probabilities (binary)
                y_test_predictions = (y_test_predictions > 0.5).astype(int)

                # Convert probabilities (multi-class)
                y_test_predictions = np.argmax(y_test_predictions, axis=1)
                ```
                """)

    # Evaluating information
    st.markdown(""" -----""")
    st.subheader("Step 5: Evaluate the model performance 📊")
    st.markdown("""
                1. Use Scikit-Learn libraries to calcualte the evaluation metrics (accuracy & F1 score)
                2. Comparing the difference between the predicted y values and true y values (y_test)

                ```python
                from sklearn.metrics import accuracy_score, f1_score

                # Accuracy
                accuracy_binary = accuracy_score(y_test, y_test_predicitions)
                # F1 Score 
                f1 = f1_score(y_test, y_test_predicitions, average='weighted', zero_division=1)
                """)