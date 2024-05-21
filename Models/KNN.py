"""
Drone BINARY Detection based on KNN, Signal Processing Letters  (SPL)    

@author: MURTIZA ALI
"""
#%%                                                     DATA LOADING 
import numpy as np
import os
from sklearn.model_selection import train_test_split

drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/MFCC 20/Drone'
no_drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/MFCC 20/No Drone'

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

#%%
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


# Define the number of folds
k = 5

# Initialize lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
conf_matrices = []
far_scores = []
mdr_scores = []

# Initialize KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=72)

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
    # Split data into train and test sets for the current fold
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    
    # Initialize and train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
    knn_classifier.fit(train_data, train_labels)
    
    # Predict test labels
    test_predictions = knn_classifier.predict(test_data)
    
    # Compute evaluation metrics for the current fold
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    far = fp / (fp + tn)
    mdr = fn / (fn + tp)
    
    # Append evaluation metrics to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    conf_matrices.append(conf_matrix)
    far_scores.append(far)
    mdr_scores.append(mdr)
    
    # Print confusion matrix for the current fold
    print(f"Confusion Matrix for Fold {fold}:")
    print(conf_matrix)
    print()

# Compute average evaluation metrics across all folds
avg_accuracy = sum(accuracy_scores) / k
avg_precision = sum(precision_scores) / k
avg_recall = sum(recall_scores) / k
avg_f1 = sum(f1_scores) / k
avg_far = sum(far_scores) / k
avg_mdr = sum(mdr_scores) / k

# Print average evaluation metrics
print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1 Score:", avg_f1)
print("Average False Alarm Rate (FAR):", avg_far)
print("Average Missed Detection Rate (MDR):", avg_mdr)

joblib.dump(knn_classifier, '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/knn_classifier_model.pkl')
