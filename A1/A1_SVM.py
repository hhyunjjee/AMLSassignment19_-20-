#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, preprocessing
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from pandas import DataFrame
import os
from sklearn import svm
from tqdm._tqdm_notebook import tqdm_notebook


# ## Data Preparation

# In[2]:


# Import landmarks
import lab2_landmarks as l2

def load_data_A1():
    # l2.extract_features_labels(Target Index)
    # Target Index from lab2_landmarks: 
    #         2 (Gender_label), 3 (Emotion_label)
    return l2.extract_features_labels(2)


# x: feature, y: label
x, y = load_data_A1()


# In[3]:


def split_data_A1(x, y):

    # Transpose of y
    yT = np.array([y, -(y - 1)]).T

    # Train-test split
    x_tr, x_te, y_tr, y_te = train_test_split(x, yT, train_size = 0.8, random_state=0)
    
    return x, yT, x_tr, x_te, y_tr, y_te

x, yT, x_tr, x_te, y_tr, y_te = split_data_A1(x,y)


# In[4]:


# Data reshaping
def reshapeX_A1(x):
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

def reshapeY_A1(y):
    return list(zip(*y))[0]

def prepare_data_A1(x_tr, x_te, y_tr, y_te):
    # Generate and reshape the train and test dataset 
    train_images = reshapeX_A1(x_tr)
    test_images = reshapeX_A1(x_te)
    train_labels = reshapeY_A1(y_tr)
    test_labels = reshapeY_A1(y_te)
    
    return train_images, test_images, train_labels, test_labels 

train_images, test_images, train_labels, test_labels = prepare_data_A1(x_tr, x_te, y_tr, y_te)


    


# ## New Test Data Generation

# In[6]:


## Import new landmarks for new test dataset
import lab2_landmarks_test as l2_test
from lab2_landmarks_test import *


# In[9]:


def load_test_data_A1():
    # l2.extract_features_labels(Target Index)
    
    # Target Index from lab2_landmarks: 
    #         2 (Gender_label), 3 (Emotion_label)

    return l2_test.extract_features_labels_test(2)



# In[45]:


# a: feature, b: label
a, b = load_test_data_A1()


# In[46]:


def newdata_prep_A1(x,y):
    yT = np.array([y, -(y - 1)]).T
    images = reshapeX_A1(x)
    labels = reshapeY_A1(yT)
    return images, labels

new_image, new_label = newdata_prep_A1(a,b)


# ## SVM Model Training using RandomSearchCV

# In[13]:


# Define the range of hyperparamter

def param_rs_linear_A1():
    
    param = {'C': stats.uniform(0.1, 10),
             'kernel': ['linear']}

    return param

def param_rs_rbf_A1():
    
    param = {'C': stats.uniform(0.1, 10),
             'gamma': stats.uniform(0.001, 0.01),
             'kernel': ['rbf']}

    return param

def param_rs_poly_A1():
    
    param = {'C': stats.uniform(0.01, 10),
             'degree': stats.uniform(1, 4),
             'kernel': ['poly']}

    return param


# In[14]:


param_rs_linear = param_rs_linear_A1()
param_rs_rbf = param_rs_rbf_A1()
param_rs_poly = param_rs_poly_A1()


# In[15]:


# Use RamdomizedSearchCV to find the best hyperparameter combination

def randomSearch_A1(X, y, param_kernel):
    #param_distributions = param_kernel 
    random_search = RandomizedSearchCV(SVC(), param_kernel, n_iter=10, n_jobs=-1, refit=True, verbose=3)
    random_search.fit(X, y)
    random_search.cv_results_

    return random_search.best_params_, random_search.best_estimator_


# In[16]:


# Obtaining optimum hyperparameters and classifier for different kernel
rs_linear_param, rs_linear = randomSearch_A1(train_images, train_labels, param_rs_linear)


# In[17]:


rs_rbf_param, rs_rbf= randomSearch_A1(train_images, train_labels, param_rs_rbf)


# In[18]:


rs_poly_param, rs_poly = randomSearch_A1(train_images, train_labels, param_rs_poly)


# In[19]:


# Display optimum hyperparameters for SVC kernel
print('\nOptimal  hyper-parameters combination (Kernel: Linear):')
print(rs_linear_param)
print('\nAccuracy Score (Kernel: Linear):')
print(rs_linear.score(test_images, test_labels))

print('\nOptimal  hyper-parameters combination (Kernel: RBF):')
print(rs_rbf_param)
print('\nAccuracy Score (Kernel: RBF):')
print(rs_rbf.score(test_images, test_labels))

print('\nOptimal  hyper-parametersand combination (Kernel: Polynomial):')
print(rs_poly_param)
print('\nAccuracy Score (Kernel: Polynomial):')
print(rs_poly.score(test_images, test_labels))


# ## Accuray score with three different kernels

# In[20]:


# Print out each kernel functions with accuracy scores
acc_scores = {'Types of Kernel' : ['Linear','RBF','Polynomial'],
           'Accuracy Score' : [rs_linear.score(test_images, test_labels), rs_rbf.score(test_images, test_labels), rs_poly.score(test_images, test_labels)]}

df_kernel_scores = DataFrame(acc_scores, columns= ['Types of Kernel', 'Accuracy Score'])
print(df_kernel_scores)


# In[21]:


def val_acc_A1(x):
    score = x.score(test_images, test_labels)
    return score

def train_acc_A1(x):
    score = x.score(train_images, train_labels)
    return score


# In[22]:


val_acc = val_acc_A1(rs_poly)
val_acc


# In[23]:


train_acc = train_acc_A1(rs_poly)
train_acc


# ## Model Prediction

# In[24]:


def prediction_A1(best_estimator):
    clf = best_estimator
    pred = clf.predict(test_images)
    return pred


# In[25]:


pred_linear = prediction_A1(rs_linear)


# In[26]:


pred_rbf = prediction_A1(rs_rbf)


# In[27]:


pred_poly = prediction_A1(rs_poly)


# ## Model Validation - Learning Curve

# In[28]:


# Plot Lenarning curve for model evaluation
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Accuray Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                                          cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes)
    
    # Compute the mean and stand deviation values for training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    
    return plt


# In[61]:


from datetime import datetime

print('Running at', str(datetime.now()))

fig, axes = plt.subplots(1, 1)

X = reshapeX_A1(x)
y = reshapeY_A1(yT)

#final_clf_lin = SVC(C= 0.1, gamma = 0.001, kernel = 'linear')
#final_clf_rbf = SVC(C= 10, gamma = 0.001, kernel='rbf')
final_clf_poly = SVC(C=9.109186022709416, degree = 3.5607644421631077, kernel='poly')


estimator = final_clf_poly

title = 'Learning Curves (Kernel = Poly)'
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

plt = plot_learning_curve(estimator, title, X, y, axes = axes, cv = cv, n_jobs = -1)
    
plt.show()
 


# ## Model Evaluation - Confusion Matrix

# In[30]:


# Define the name of each class
class_names = np.array(['Male', 'Female'])

# Plt confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Non-normalized Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(test_labels, pred_poly)
    classes = class_names
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Non-normalized Confusion Matrix')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation=None, cmap=cmap)
    
    print(cm)
    ax.figure.colorbar(im, ax=ax)

    for i in range(len(ax.yaxis.get_major_ticks())):
        ax.yaxis.get_major_ticks()[i].label1.set_visible(False)
        
    for i in [2, 6]:
        ax.yaxis.get_major_ticks()[i].label1.set_visible(True)
        
    ax.set(
        xticks = np.arange(cm.shape[1]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=[0, 0, classes[0], 0, 0, 0, classes[1], 0, 0],
       title=title,
       ylabel='True label',
       xlabel='Predicted label'
    )
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(test_images, pred_poly, classes=class_names,
                      title='Non-normalized Confusion Matrix')

# Plot normalized confusion matrix
plot_confusion_matrix(test_images, pred_poly, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Model Testing on New Dataset

# In[58]:


def new_prediction_A1(best_estimator):
    clf = best_estimator
    pred = clf.predict(new_image)
    return pred


# In[60]:


new_pred = new_prediction_A1(rs_poly)
new_acc = accuracy_score(new_label, new_pred)
new_acc


# In[ ]:




