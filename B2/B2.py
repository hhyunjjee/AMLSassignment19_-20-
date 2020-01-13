#!/usr/bin/env python
# coding: utf-8

# In[201]:


# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score,recall_score, precision_score,f1_score
import pandas as pd
from sklearn.datasets import load_iris ##
import os
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
from tqdm._tqdm_notebook import tqdm_notebook
import pandas as pd
import cv2
from keras.models import load_model


# ## Directory

# In[220]:


base_dir_b2= ("/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B2")
os.chdir(base_dir_b2)

dataset_dir_b2 = os.path.join(base_dir_b2, 'img')
labels_filename_b2 = os.path.join(base_dir_b2, 'labels.csv')

test_base_dir_b2 = ("/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/Dataset/cartoon_set_test")
test_dataset_dir_b2 = os.path.join(test_base_dir_b2, 'img')
test_labels_filename_b2 = os.path.join(test_base_dir_b2, 'labels_test.csv')



# ## Data Preparation

# In[190]:


def data_frame_B2 (labels_filename_b2):
    df_b2 = pd.read_csv(labels_filename_b2)
    df_b2.columns=['original']
    df_b2["file_name"] = df_b2['original'].str.split("\t").str[3]
    df_b2["eyecolour_label"] = df_b2['original'].str.split("\t").str[1]
    del df_b2['original']
    eye_colour = []
    
    for eye in df_b2.eyecolour_label:
        if eye == '0':
            eye_colour.append("Brown")
        elif eye == '1':
            eye_colour.append("Blue")
        elif eye == '2':
            eye_colour.append("Green")
        elif eye == '3':
            eye_colour.append("Gray")
        else:
            eye_colour.append("Black")
    
    df_b2['eye_colour'] = eye_colour
    

    return df_b2
    


# In[191]:


df_b2 = data_frame_B2(labels_filename_b2)
df_b2


# ## New Test Data Preparation

# In[221]:


def test_df_B2 (test_labels_filename_b2):
    df_b2 = pd.read_csv(test_labels_filename_b2)
    df_b2.columns=['original']
    df_b2["file_name"] = df_b2['original'].str.split("\t").str[3]
    df_b2["eyecolour_label"] = df_b2['original'].str.split("\t").str[1]
    del df_b2['original']
    eye_colour = []
    
    for eye in df_b2.eyecolour_label:
        if eye == '0':
            eye_colour.append("Brown")
        elif eye == '1':
            eye_colour.append("Blue")
        elif eye == '2':
            eye_colour.append("Green")
        elif eye == '3':
            eye_colour.append("Gray")
        else:
            eye_colour.append("Black")
    
    df_b2['eye_colour'] = eye_colour
    

    return df_b2

new_test_b2 = test_df_B2 (test_labels_filename_b2)
new_test_b2


# In[192]:


from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.optimizers import adam
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# ## Model Evaluation

# In[193]:


def split_data_B2(df):

    train_data, test_data = train_test_split(df, random_state=0)
    
    return train_data, test_data


# In[194]:


train_data, test_data = split_data_B2(df_b2)


# In[146]:



train_b2, test_b2 = train_test_split(df_b2,train_size=0.8, random_state=0)
    
# Setup the data generator
data_generator_b2 = ImageDataGenerator(
    rescale = 1./255.,
    validation_split = 0.25,
    horizontal_flip=True,
    vertical_flip=True
    
)
    
# Get batches of training dataset from the dataframe
print("Training Dataset Preparataion: ")
train_generator_b2 = data_generator_b2.flow_from_dataframe(
        dataframe = train_b2, directory = dataset_dir_b2,
        x_col = "file_name", y_col = "eyecolour_label",
        class_mode = 'categorical', target_size = (64,64),
        batch_size = 128, subset = 'training')
    
    # Get batches of validation dataset from the dataframe
print("\nValidation Dataset Preparataion: ")
validation_generator_b2 = data_generator_b2.flow_from_dataframe(
        dataframe = train_b2, directory = dataset_dir_b2,
        x_col = "file_name", y_col = "eyecolour_label",
        class_mode = 'categorical', target_size = (64,64),
        batch_size = 128, subset = 'validation')
    
    
    


# In[219]:


# # starting point 
model2= models.Sequential()

# Add first convolutional block
model2.add(Conv2D(16, (3, 3), activation='relu',input_shape=(64,64,3)))
model2.add(MaxPooling2D((2, 2), padding = 'same'))

# second block
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2), padding = 'same'))

# third block
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2), padding = 'same'))

# fourth block
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2), padding = 'same'))


model2.add(Flatten())
model2.add(Dense(5, activation='softmax'))

model2.summary()


# In[196]:


# # use early stopping to optimally terminate training through callbacks
# # from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# # # save best model automatically
# mc= ModelCheckpoint('/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B2/model_B2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# cb_list=[es,mc]


# compile model 
opt_adam = optimizers.Adam(lr=0.001)
model2.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])


# In[197]:


history_b2 = model2.fit_generator(
        train_generator_b2,
        epochs=25,
        steps_per_epoch=train_generator_b2.samples // 128,
        validation_data=validation_generator_b2,
        validation_steps=validation_generator_b2.samples // 128)
        #callbacks=cb_list)


# In[217]:


# plot training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history_b2.history['accuracy'])
plt.plot(history_b2.history['val_accuracy'])
#plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("model_B1_1.png", dpi=300)


# In[218]:


# plot loss for trainig and validation
import matplotlib.pyplot as plt
plt.plot(history_b2.history['loss'])
plt.plot(history_b2.history['val_loss'])
#plt.ylim([.5,1.1])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("model_B2_2.png", dpi=300)


# ## Model Evaluation

# In[215]:


print("Test Dataset Preparataion")
test_data_generator_b2 = ImageDataGenerator(rescale=1./255)
test_generator_b2 = test_data_generator_b2.flow_from_dataframe(
        dataframe = test_b2, directory = dataset_dir_b2,
        x_col = "file_name", y_col = "eyecolour_label",
        class_mode = 'categorical', target_size = (64,64),
        batch_size =1, shuffle = False)

file_names_b2 = test_generator_b2.filenames
sample_size_b2 = len(file_names_b2)

model_pred_test_b2 = model2.predict_generator(test_generator_b2, sample_size_b2)

model_path_b2 = '/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B2/model_B2.h5'
saved_model_b2 = load_model(model_path_b2)

model_pred_test_b2 = saved_model_b2.predict_generator(test_generator_b2, sample_size_b2)



# ## Confusion Matrix

# In[206]:


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_names = np.array(['Brown', 'Blue','Green','Gray','Black'])

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
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'



    cm= confusion_matrix(true_test_b2, pred_test_b2)
    classes = class_names
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

pred_test_b2  = np.argmax(model_pred_test_b2, axis = 1)
true_test_b2 = np.array(test_generator_b2.classes)
# Plot non-normalized confusion matrix
plot_confusion_matrix(true_test_b2, pred_test_b2, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(true_test_b2, pred_test_b2, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Validation Accuravy

# In[207]:


acc_comp_b2 = accuracy_score(true_test_b2, pred_test_b2)
rec_comp_b2 = recall_score(true_test_b2, pred_test_b2, pos_label = 'positive', average ='macro')
prec_comp_b2 = precision_score(true_test_b2, pred_test_b2, pos_label = 'positive', average ='macro')
f1_comp_b2 = f1_score(true_test_b2, pred_test_b2, pos_label = 'positive', average ='macro')
print("Accuracy :" + str(acc_comp_b2))
print("Precision :" + str(prec_comp_b2))
print("Recall :" + str(rec_comp_b2))
print("F1 Score :" + str(f1_comp_b2))


# In[ ]:


pred_test_b2  = np.argmax(model_pred_test_b2, axis = 1)
true_test_b2 = np.array(test_generator_b2.classes)


# ## Train Accuracy

# In[216]:


score_generator_b2 = data_generator_b2.flow_from_dataframe(
        dataframe = train_data, directory = dataset_dir_b2,
        x_col = "file_name", y_col = "eyecolour_label",
        class_mode = 'categorical', target_size = (64,64),
        batch_size =1, shuffle = False)

train_metric = saved_model_b2.evaluate_generator(score_generator_b2, steps = validation_generator_b2.samples // 32, verbose=1)

print('Train loss: '+ str(train_metric[0]))
print('Train Accuracy: '+ str(train_metric[1]))
      
      
      
      
      


# ## Test Accuracy

# In[226]:


test_score_generator_b2 = data_generator_b2.flow_from_dataframe(
        dataframe = new_test_b2, directory = test_dataset_dir_b2,
        x_col = "file_name", y_col = "eyecolour_label",
        class_mode = 'categorical', target_size = (64,64),
        batch_size =1, shuffle = False)

new_test_metric = saved_model_b2.evaluate_generator(test_score_generator_b2, steps = validation_generator_b2.samples // 32, verbose=1)

print('Test loss: '+ str(new_test_metric[0]))
print('Test Accuracy: '+ str(new_test_metric[1]))
      
      


# In[ ]:




