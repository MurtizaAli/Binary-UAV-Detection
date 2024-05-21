
"""
Drone BINARY Detection based on 2D-CNN, Signal Processing Letters  (SPL)    

@author: MURTIZA ALI
"""

#%%                                                     DATA LOADING 
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
def set_specific_gpu(ID):
      gpus_all_physical_list = tf.config.list_physical_devices(device_type='GPU')  
      # print(gpus_all_physical_list)
      tf.config.set_visible_devices(gpus_all_physical_list[ID], 'GPU')
set_specific_gpu(1)

drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/MFCC 20/Drone'
no_drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/2D Features/MFCC 20/No Drone'

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

drone_labels = np.ones(len(drone_data))  # Assign label 1 for drone class
no_drone_labels = np.zeros(len(no_drone_data))  # Assign label 0 for no-drone class
data = np.concatenate((drone_data, no_drone_data), axis=0)
labels = np.concatenate((drone_labels, no_drone_labels), axis=0)

# Shuffle the data and labels
random_indices = np.random.permutation(len(data))
data = data[random_indices]
labels = labels[random_indices]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=72)

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
del data
del labels
#%%


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Conv2D(32, (3, 3), activation=None, input_shape=(20,69,1), padding='same')) #bases on the size of the feature used
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3, 3), activation=None, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))


model.add(Conv2D(64, (3, 3), activation=None, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))
# model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(64, activation=None))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(1, activation=None))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #use binary cross entropy
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
num_epochs = 35
num_batch_size = 64

history = model.fit(train_data, train_labels, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.3, verbose=2, callbacks=[early_stopping])

#%% 
#%% Save and load the saved model
model.save("/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/2D_CNN_saved_model")
# del train_data
# log_eps = 1e-6
# model = tf.keras.models.load_model("/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/WST_CNN_saved_model")
#%%                                                            EVALUATION OF THE MODEL
del train_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
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
