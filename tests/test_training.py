# In test_training.py

from unittest.mock import patch
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import preprocess_data
from models import create_model, focal_loss_with_class_weights
from tensorflow.keras.optimizers import Adam

@patch('preprocessing.pd.read_csv')
def test_training_pipeline(mock_read_csv):
    # Mocking pd.read_csv to return dummy DataFrame for test
    mock_read_csv.return_value = pd.DataFrame({
        'ecg_id': [1],
        'scp_codes': [{'NORM': 1}],
        'age': [60],
        'sex': [1],
        'strat_fold': [1],
        'filename_lr': ['00001_lr'],
        'filename_hr': ['00001_hr']
    })

    path = 'dataset/ptb-xl/1.0.3/'
    sampling_rate = 500
    ptbxl_database_path = path + 'ptbxl_database.csv'
    scp_statements_path = path + 'scp_statements.csv'
    output_path = 'preprocessed_dataset_test.csv'

    # Preprocess the data
    X_ecg, X_features, data, Y = preprocess_data(ptbxl_database_path, scp_statements_path, output_path, sampling_rate, path)


    # Convert data types
    X_ecg = X_ecg.astype(np.float32)
    X_features = X_features.astype(np.float32)
    Y = Y.astype(np.float32)

    # Split the data for training, validation, and test
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

    # Handle class imbalance
    class_weights_list = []
    for i in range(y_train.shape[1]):
        cw = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train.iloc[:, i]),
            y=y_train.iloc[:, i]
        )
        class_weights_list.append(cw[1] / cw[0])

    class_weight_values = np.array(class_weights_list)
    class_weight_tensor = tf.constant(class_weight_values, dtype=tf.float32)

    # Best hyperparameters from previous hyperparameter tuning
    hparams = {
        'num_res_blocks': 1,  # Reduce for quicker tests
        'filters': [64],      # Reduced filters for testing
        'kernel_sizes': [3],
        'dropout_rate': 0.25,
        'lstm_units': 64,     # Smaller LSTM units for quicker tests
        'dense_units': 128,
        'meta_dense_units': 16,
        'combined_dense_units': 64,
        'learning_rate': 1e-4,
        'alpha': 0.25,
        'gamma': 2.0
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
                  loss=lambda y_true, y_pred: focal_loss_with_class_weights(y_true, y_pred, hparams['alpha'], hparams['gamma'], class_weight_tensor),
                  metrics=[AUC(name='val_auc')])

    # Shortened training for tests
    early_stopping = EarlyStopping(monitor='val_auc', patience=2, mode='max', restore_best_weights=True)

    model.fit([X_ecg_train, X_features_train], y_train.values,
              validation_data=([X_ecg_val, X_features_val], y_val.values),
              epochs=5,  # Reduced epochs for quicker testing
              batch_size=32,
              callbacks=[early_stopping])

    # Evaluate on test set
    y_pred = model.predict([X_ecg_test, X_features_test])
    
    # Use metrics for evaluation
    precision = precision_score(y_test, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    recall = recall_score(y_test, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    f1 = f1_score(y_test, (y_pred > 0.5).astype(int), average='macro', zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred, average='macro')
    subset_accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    hamming = hamming_loss(y_test, (y_pred > 0.5).astype(int))

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("AUC-ROC:", auc_roc)
    print("Subset Accuracy:", subset_accuracy)
    print("Hamming Loss:", hamming)

if __name__ == "__main__":
    test_training_pipeline()
