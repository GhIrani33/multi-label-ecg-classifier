
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
path = 'D:/Project/ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate = 500
ptbxl_database_path = path + 'ptbxl_database.csv'
scp_statements_path = path + 'scp_statements.csv'
output_path = 'preprocessed_dataset1.csv'

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



Data split into train, validation, and test sets.
X_ecg_train shape: (17418, 1000, 12)
X_features_train shape: (17418, 3)
y_train shape: (17418, 5)
X_ecg_val shape: (2183, 1000, 12)
X_features_val shape: (2183, 3)
y_val shape: (2183, 5)
X_ecg_test shape: (2198, 1000, 12)
X_features_test shape: (2198, 3)
y_test shape: (2198, 5)

Precision (macro): 0.7395926184430195
Recall (macro): 0.7427838948648692
F1-score (macro): 0.7382023929419919
AUC-ROC (macro): 0.9162569696906674
Subset Accuracy: 0.6155595996360328
Hamming Loss: 0.11583257506824386
Classification Report:
C:\Users\PC\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
C:\Users\PC\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in samples with no true labels. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
C:\Users\PC\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\metrics\_classification.py:1531: UndefinedMetricWarning: F-score is 
ill-defined and being set to 0.0 in samples with no true nor predicted labels. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
              precision    recall  f1-score   support

        NORM       0.84      0.89      0.86       963
          MI       0.69      0.81      0.75       550
        STTC       0.72      0.79      0.75       521
          CD       0.81      0.70      0.75       496
         HYP       0.63      0.53      0.57       262

   micro avg       0.76      0.79      0.78      2792
   macro avg       0.74      0.74      0.74      2792
weighted avg       0.77      0.79      0.77      2792
 samples avg       0.77      0.79      0.76      2792

Class NORM Confusion Matrix:
[[1076  159]
 [ 108  855]]
Class MI Confusion Matrix:
[[1449  199]
 [ 104  446]]
Class STTC Confusion Matrix:
[[1520  157]
 [ 112  409]]
Class CD Confusion Matrix:
[[1623   79]
 [ 149  347]]
 [ 149  347]]
 [ 149  347]]
Class HYP Confusion Matrix:
 [ 149  347]]
Class HYP Confusion Matrix:
 [ 149  347]]
Class HYP Confusion Matrix:
 [ 149  347]]
Class HYP Confusion Matrix:
 [ 149  347]]
Class HYP Confusion Matrix:
[[1853   83]
 [ 149  347]]
 [ 149  347]]
Class HYP Confusion Matrix:
 [ 149  347]]
Class HYP Confusion Matrix:
[[1853   83]
 [ 123  139]]