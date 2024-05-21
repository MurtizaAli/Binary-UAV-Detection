"""
Drone BINARY Detection based on SVM, Signal Processing Letters  (SPL)    

@author: MURTIZA ALI
"""
#%%                                                     DATA LOADING 
import numpy as np
import os
from sklearn.model_selection import train_test_split

drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/PSD/Drone'
no_drone_data_dir = '/home/user/Desktop/Drone Detection Pre-processing Data/Binary Drone Detection/Dataset/min30_2_pul10/1D Features/PSD/No Drone'


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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define the number of folds
k = 5

# Initialize lists to store evaluation metrics for each kernel
kernel_eval_metrics = {}

# Specify the kernels to evaluate
kernels = ['linear', 'poly', 'rbf']

# Initialize KFold object
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Iterate over each kernel
for kernel in kernels:
    # Initialize lists to store evaluation metrics for the current kernel
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    conf_matrices = []
    far_scores = []
    mdr_scores = []

    # Iterate over each fold
    for train_index, test_index in kf.split(data):
        # Split data into train and test sets for the current fold
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create and train SVM classifier with the current kernel
        svm_model = SVC(kernel=kernel)
        svm_model.fit(train_data, train_labels)

        # Predict test labels
        test_predictions = svm_model.predict(test_data)

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

    # Compute average evaluation metrics across all folds for the current kernel
    avg_accuracy = sum(accuracy_scores) / k
    avg_precision = sum(precision_scores) / k
    avg_recall = sum(recall_scores) / k
    avg_f1 = sum(f1_scores) / k
    avg_far = sum(far_scores) / k
    avg_mdr = sum(mdr_scores) / k

    # Store average evaluation metrics for the current kernel
    kernel_eval_metrics[kernel] = {
        'Accuracy': avg_accuracy,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1 Score': avg_f1,
        'False Alarm Rate (FAR)': avg_far,
        'Missed Detection Rate (MDR)': avg_mdr
    }

# Print evaluation metrics for each kernel
for kernel, metrics in kernel_eval_metrics.items():
    print(f"Results for SVM with {kernel} kernel:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

