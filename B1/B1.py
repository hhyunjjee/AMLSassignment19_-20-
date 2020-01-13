#!/usr/bin/env python
# coding: utf-8

# In[45]:


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

# In[50]:


base_dir = ("/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B1")
os.chdir(base_dir)

dataset_dir = os.path.join(base_dir, 'img')
labels_filename = os.path.join(base_dir, 'labels.csv')


# ## Data Preparation

# In[13]:


def data_frame_B1 (labels_filename):
    df = pd.read_csv(labels_filename)
    df.columns=['original']
    df["file_name"] = df['original'].str.split("\t").str[3]
    df["faceShape_label"] = df['original'].str.split("\t").str[2]
    del df['original']
    return df
    
    


# In[14]:


df = data_frame_B1(labels_filename)
df


# In[15]:


from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.optimizers import adam
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# In[16]:


def split_data_B1(df):

    train_data, test_data = train_test_split(df, random_state=0)
    
    return train_data, test_data


# In[17]:


train_data, test_data = split_data_B1(df)


# In[57]:


# Setup the data generator
data_generator = ImageDataGenerator(
    rescale = 1./255.,
    validation_split = 0.25,
    horizontal_flip=True,
    vertical_flip=True
    
)
    
# Get batches of training dataset from the dataframe
print("Training Dataset Preparation")
train_generator = data_generator.flow_from_dataframe(
        dataframe = train_data, directory = dataset_dir,
        x_col = "file_name", y_col = "faceShape_label",
        class_mode = 'categorical', target_size = (32,32),
        batch_size = 64, subset = 'training')
    
    # Get batches of validation dataset from the dataframe
print("\nValidation Dataset Preparation")
validation_generator = data_generator.flow_from_dataframe(
        dataframe = train_data, directory = dataset_dir,
        x_col = "file_name", y_col = "faceShape_label",
        class_mode = 'categorical', target_size = (32,32),
        batch_size = 64, subset = 'validation')
    
    
    


# In[ ]:





# ## CNN Architecture

# In[58]:


# starting point 
model= models.Sequential()

# Add first convolutional block
model.add(Conv2D(24, (3, 3), activation='relu', padding='same',input_shape=(32,32,3)))

model.add(MaxPooling2D((2, 2), padding='same'))

# second block
model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
# third block
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
# fourth block
model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))


model.add(Flatten())
model.add(Dense(5, activation='softmax'))




# Show a summary of the model. Check the number of trainable parameters
model.summary()



# In[59]:


# use early stopping to optimally terminate training through callbacks
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# # save best model automatically
# mc= ModelCheckpoint('/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B1/model_B1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# cb_list=[es,mc]


# compile model 
opt_adam = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])


# In[60]:


history = model.fit_generator(
        train_generator,
        epochs=25,
        steps_per_epoch=train_generator.samples // 64,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 64)
#         callbacks=cb_list)


# In[61]:


# plot accuracy score for trainig and validation
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.ylim([.5,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("model_B1_1.png", dpi=300)


# In[62]:


# plot loss for trainig and validation
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.ylim([.5,1.1])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("model_B1_2.png", dpi=300)


# ## Model Evaluation 
# 

# In[92]:


from keras.models import load_model

print("Test Dataset Preparataion")
test_data_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_data_generator.flow_from_dataframe(
        dataframe = train_data, directory = dataset_dir,
        x_col = "file_name", y_col = "faceShape_label",
        class_mode = 'categorical', target_size = (32,32),
        batch_size =1, shuffle = False)

file_names = test_generator.filenames
sample_size = len(file_names)

model_path = '/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/B1/model_B1.h5'
saved_model = load_model(model_path)

model_pred_test = saved_model.predict_generator(test_generator, sample_size)


# ## Confusion Matrix

# In[93]:


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class_names = np.array(['FaceShape0', 'FaceShape1','FaceShape2','FaceShape3','FaceShape4'])

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



    cm = confusion_matrix(true_test, pred_test)
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
       #xticklabels=classes, yticklabels=[0, 0, classes[0], 0, 0, 0, classes[1], 0, 0],
       xticklabels=classes, yticklabels=classes,
        #yticklabels=[0, 0, classes[0], 0, 0, 0, classes[1], 0, 0],
       
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

pred_test = np.argmax(model_pred_test, axis = 1)
true_test = np.array(test_generator.classes)
# Plot non-normalized confusion matrix
plot_confusion_matrix(true_test, pred_test, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(true_test, pred_test, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ## Validation Accuracy

# In[94]:


acc_comp = accuracy_score(true_test, pred_test)
rec_comp = recall_score(true_test, pred_test, pos_label = 'positive', average ='macro')
prec_comp = precision_score(true_test, pred_test, pos_label = 'positive', average ='macro')
f1_comp = f1_score(true_test, pred_test, pos_label = 'positive', average ='macro')
print("Accuracy :" + str(acc_comp))
print("Precision :" + str(prec_comp))
print("Recall :" + str(rec_comp))
print("F1 Score :" + str(f1_comp))


# ## Train Accuracy

# In[95]:


score_generator = data_generator.flow_from_dataframe(
        dataframe = train_data, directory = dataset_dir,
        x_col = "file_name", y_col = "faceShape_label",
        class_mode = 'categorical', target_size = (32,32),
        batch_size =1, shuffle = False)

train_metric = saved_model.evaluate_generator(score_generator, steps = validation_generator.samples // 32, verbose=1)

print('Train loss: '+ str(train_metric[0]))
print('Train Accuracy: '+ str(train_metric[1]))
      
      
      
      
      


# ## New Test Dataset Preparation

# In[96]:


test_labels_filename_b1


# In[97]:



test_base_dir_b1 = ("/Users/Hyunjee/Desktop/AMLS_19-20_HYUNJEE_KIM_SN16075203/Dataset/cartoon_set_test")
test_dataset_dir_b1 = os.path.join(test_base_dir_b1, 'img')
test_labels_filename_b1 = os.path.join(test_base_dir_b1, 'labels_test.csv')


# In[98]:


def test_df_B1 (test_labels_filename):
    df_b1 = pd.read_csv(test_labels_filename)
    df_b1.columns=['original']
    df_b1["file_name"] = df_b1['original'].str.split("\t").str[3]
    df_b1["faceShape_label"] = df_b1['original'].str.split("\t").str[2]
    del df_b1['original']
    return df_b1

new_test_b1 = test_df_B1 (test_labels_filename_b1)
new_test_b1


# ## Test Accuracy

# In[99]:


test_score_generator = data_generator.flow_from_dataframe(
        dataframe = new_test_b1, directory = test_dataset_dir_b1,
        x_col = "file_name", y_col = "faceShape_label",
        class_mode = 'categorical', target_size = (32,32),
        batch_size =1, shuffle = False)

new_test_metric = saved_model.evaluate_generator(test_score_generator, steps = validation_generator.samples // 64, verbose=1)

print('Test loss: '+ str(new_test_metric[0]))
print('Test Accuracy: '+ str(new_test_metric[1]))
      
      


# In[ ]:





# In[ ]:





# In[ ]:




