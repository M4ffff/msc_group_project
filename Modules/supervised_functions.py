import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import make_classification,make_blobs
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder



def plot_decision_boundary(X, y, model, alpha=0.8, cmap='viridis'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.figure(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=alpha)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k')
    ax.xlabel('Feature 1')
    ax.ylabel('Feature 2')
    ax.title('Decision Boundary')
    st.pyplot(fig)



def plot_decision_boundary2(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.figure(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm', alpha=0.8)
    ax.title('Decision Boundary')
    st.pyplot(fig)



def plot_decision_boundary_with_hyperplane(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='k', linestyles='--', levels=[-1, 0, 1], linewidths=1)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50, label='Data Points')

    w = model.coef_[0]
    b = model.intercept_[0]
    x_values = np.linspace(x_min, x_max, 100)
    y_values = -(w[0] * x_values + b) / w[1]

    ax.plot(x_values, y_values, color='black', linestyle='-', linewidth=2, label='Hyperplane')

    margin = 1 / np.sqrt(np.sum(w ** 2))
    y_values_upper = y_values + margin * (w[1] / np.linalg.norm(w))
    y_values_lower = y_values - margin * (w[1] / np.linalg.norm(w))

    ax.plot(x_values, y_values_upper, color='gray', linestyle='--', linewidth=1, label='Margin')
    ax.plot(x_values, y_values_lower, color='gray', linestyle='--', linewidth=1)

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            facecolors='none', edgecolors='black', s=120, linewidths=1.5, label='Support Vectors')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary with Hyperplane and Support Vectors')
    ax.legend()

    st.pyplot(fig)



