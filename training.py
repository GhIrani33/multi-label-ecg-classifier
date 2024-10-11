
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from sklearn.utils import class_weight
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score, hamming_loss,
                             classification_report, multilabel_confusion_matrix)
import os

# Import custom modules
from preprocessing import preprocess_data
from models import create_model, focal_loss_with_class_weights

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Specify the paths to the dataset and files
path = 'dataset/ptb-xl/1.0.3/'
ptbxl_database_path = path + 'ptbxl_database.csv'
scp_statements_path = path + 'scp_statements.csv'
output_path = 'preprocessed_dataset.csv'


X_ecg, X_features, data, Y = preprocess_data(ptbxl_database_path, scp_statements_path, output_path, sampling_rate, path)


# Convert data types
X_ecg = X_ecg.astype(np.float32)
X_features = X_features.astype(np.float32)
Y = Y.astype(np.float32)


# Split data according to dataset recommendations
train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
validation_fold = 9
test_fold = 10

train_indices = data['strat_fold'].isin(train_folds)
val_indices = data['strat_fold'] == validation_fold
test_indices = data['strat_fold'] == test_fold

X_ecg_train = X_ecg[train_indices]
X_features_train = X_features[train_indices]
y_train = Y[train_indices]

X_ecg_val = X_ecg[val_indices]
X_features_val = X_features[val_indices]
y_val = Y[val_indices]

X_ecg_test = X_ecg[test_indices]
X_features_test = X_features[test_indices]
y_test = Y[test_indices]


print("Data split into train, validation, and test sets.")
# Print the shapes of the split data
print("X_ecg_train shape:", X_ecg_train.shape)
print("X_features_train shape:", X_features_train.shape)
print("y_train shape:", y_train.shape)
print("X_ecg_val shape:", X_ecg_val.shape)
print("X_features_val shape:", X_features_val.shape)
print("y_val shape:", y_val.shape)
print("X_ecg_test shape:", X_ecg_test.shape)
print("X_features_test shape:", X_features_test.shape)
print("y_test shape:", y_test.shape)

# Handling Class Imbalance
class_weights_list = []
for i in range(y_train.shape[1]):
    cw = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train.iloc[:, i]),
        y=y_train.iloc[:, i]
    )
    class_weights_list.append(cw[1] / cw[0])

# Convert class weights to a tensor
class_weight_values = np.array(class_weights_list)
class_weight_tensor = tf.constant(class_weight_values, dtype=tf.float32)

# Best hyperparameters from previous hyperparameter tuning
hparams = {
    'num_res_blocks': 2,
    'filters': [256, 128],
    'kernel_sizes': [3, 7],
    'dropout_rate': 0.2532911469635144,
    'lstm_units': 128,
    'dense_units': 256,
    'meta_dense_units': 16,
    'combined_dense_units': 128,
    'learning_rate': 0.00046744485468784396,
    'alpha': 0.434606549138721,
    'gamma': 2.1350570960531274
}

# Define input shapes
input_shape_ecg = (X_ecg_train.shape[1], X_ecg_train.shape[2])
input_shape_meta = (X_features_train.shape[1],)
num_classes = y_train.shape[1]

# Build the model
model = create_model(hparams, input_shape_ecg, input_shape_meta, num_classes)

# Compile the model
optimizer = Adam(learning_rate=hparams['learning_rate'])
model.compile(optimizer=optimizer,
              loss=lambda y_true, y_pred: focal_loss_with_class_weights(y_true, y_pred,
                                                                        hparams['alpha'], hparams['gamma'],
                                                                        class_weight_tensor),
              metrics=[AUC(name='val_auc')])

# Callbacks
early_stopping = EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='Final-ECG-Model.h5',
    monitor='val_auc',
    mode='max',
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max')

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Training
history = model.fit(
    [X_ecg_train, X_features_train],
    y_train.values,
    validation_data=([X_ecg_val, X_features_val], y_val.values),
    epochs=100,
    batch_size=64,
    callbacks=callbacks
)

# Evaluate on test set
y_pred = model.predict([X_ecg_test, X_features_test])

# Determine optimal thresholds
optimal_thresholds = []
for i in range(y_test.shape[1]):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test.iloc[:, i], y_pred[:, i])
    f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
    idx = np.argmax(f1_scores)
    if idx < len(thresholds):
        optimal_threshold = thresholds[idx]
    else:
        optimal_threshold = 0.5
    optimal_thresholds.append(optimal_threshold)

y_pred_binary = np.zeros_like(y_pred)
for i in range(y_pred.shape[1]):
    y_pred_binary[:, i] = (y_pred[:, i] >= optimal_thresholds[i]).astype(int)

# Compute performance metrics
precision = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred, average='macro')
subset_accuracy = accuracy_score(y_test, y_pred_binary)
hamming = hamming_loss(y_test, y_pred_binary)

print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("AUC-ROC (macro):", auc_roc)
print("Subset Accuracy:", subset_accuracy)
print("Hamming Loss:", hamming)

print("Classification Report:")
print(classification_report(y_test, y_pred_binary, target_names=y_test.columns))

conf_matrix = multilabel_confusion_matrix(y_test, y_pred_binary)
for idx, matrix in enumerate(conf_matrix):
    print(f"Class {y_test.columns[idx]} Confusion Matrix:")
    print(matrix)

# Save the final model
model.save('Final-ECG-Model.keras')

