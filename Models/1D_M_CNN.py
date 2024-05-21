"""
Drone BINARY Detection based on 1D-M-CNN, Signal Processing Letters  (SPL)    

@author: MURTIZA ALI
"""

#%%                                          DATA LOADING 
import tensorflow as tf
def set_specific_gpu(ID):
      gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')  
      # print(gpus_all_physical_list)
      tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')
set_specific_gpu(1)
                                                   
import numpy as np
import os
from sklearn.model_selection import train_test_split

drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/1D_M_CNN/Drone'
no_drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/1D_M_CNN/No Drone'

# Function to load numpy data from a directory
def load_numpy_data(directory):
    data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Load numpy file
        numpy_data = np.load(file_path)
        data.append(numpy_data)
    return np.array(data)

drone_data = load_numpy_data(drone_data_dir)
no_drone_data = load_numpy_data(no_drone_data_dir)

# Create labels for drone and no-drone classes
drone_labels = np.ones(len(drone_data))  # Assign label 1 for drone class
no_drone_labels = np.zeros(len(no_drone_data))  # Assign label 0 for no-drone class

data = np.concatenate((drone_data, no_drone_data), axis=0)
labels = np.concatenate((drone_labels, no_drone_labels), axis=0)

# Shuffle the data and labels
random_indices = np.random.permutation(len(data))
data = data[random_indices]
labels = labels[random_indices]

# Split data into training, validation, and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=72)

# Print shapes of datasets for verification
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
del data
del labels
#%%                                                         TRAINING THE MODEL
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten, Reshape   #add this
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, GlobalMaxPool1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv1D(16, 3, activation='relu', input_shape=(128, 1), padding='valid'))  #bases on the size of the feature used
model.add(Conv1D(4, 3, activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
#%%

opt = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
num_epochs = 30
num_batch_size = 64

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

history = model.fit(train_data, train_labels, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.3, verbose=2, callbacks=[early_stopping])

#%%   
#%% Save and load the saved model
model.save("/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/1D_F_CNN_saved_model")
# del train_data
# log_eps = 1e-6
# model = tf.keras.models.load_model("/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/WST_CNN_saved_model")                                                          EVALUATION OF THE MODEL
#%%
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Accuracy:", test_accuracy)

# Predict test labels
test_predictions = model.predict(test_data)
test_predictions = (test_predictions > 0.5).astype(int)

# Compute precision, recall, F1-score
precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# Compute confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

from sklearn.metrics import roc_curve, auc
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
# Compute AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

tn, fp, fn, tp = conf_matrix.ravel()
# Compute false alarm rate (FAR)
far = fp / (fp + tn)
# Compute missed detection rate (MDR)
mdr = fn / (fn + tp)

print("False Alarm Rate (FAR):", far)
print("Missed Detection Rate (MDR):", mdr)
