
"""
Evaluating models on different SNRs

@author: user
"""

#%% Testing WST CNN
import numpy as np
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
def set_specific_gpu(ID):
      gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')  
      # print(gpus_all_physical_list)
      tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')
set_specific_gpu(1)

drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Testing at various SNRs/2D Features/MFCC 2D/-30/drone'
no_drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Testing at various SNRs/2D Features/MFCC 2D/-30/no_drone'

# Function to load numpy data from a directory
def load_numpy_data(directory):
    data = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
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

# log_eps = 1e-6
model = tf.keras.models.load_model("/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/2D_CNN_saved_model")
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(data, labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict test labels
test_predictions = model.predict(data)
test_predictions = (test_predictions > 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
# Compute precision, recall, F1-score
precision = precision_score(labels, test_predictions)
recall = recall_score(labels, test_predictions)
f1 = f1_score(labels, test_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)


tn, fp, fn, tp = conf_matrix.ravel()
# Compute false alarm rate (FAR)
far = fp / (fp + tn)
# Compute missed detection rate (MDR)
mdr = fn / (fn + tp)

print("False Alarm Rate (FAR):", far)
print("Missed Detection Rate (MDR):", mdr)