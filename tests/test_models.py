import unittest
import numpy as np
import tensorflow as tf
from models import create_model, focal_loss_with_class_weights

class TestModels(unittest.TestCase):

    def test_create_model(self):
        # Set hyperparameters for testing
        hparams = {
            'num_res_blocks': 2,
            'filters': [256, 128],
            'kernel_sizes': [3, 7],
            'dropout_rate': 0.25,
            'lstm_units': 128,
            'dense_units': 256,
            'meta_dense_units': 16,
            'combined_dense_units': 128
        }
        input_shape_ecg = (1000, 12)  # ECG input shape (1000 timesteps, 12 leads)
        input_shape_meta = (3,)  # Metadata input shape (3 features: age, sex)
        num_classes = 5  # Number of output classes

        # Create the model using the provided function
        model = create_model(hparams, input_shape_ecg, input_shape_meta, num_classes)

        # Test model input and output shapes
        self.assertEqual(model.input_shape, [(None, 1000, 12), (None, 3)])  # Check input shapes
        self.assertEqual(model.output_shape, (None, 5))  # Check output shape for multi-label classification

    def test_focal_loss_with_class_weights(self):
        # Sample data for loss function testing
        y_true = tf.constant([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1, 0.8, 0.2, 0.7], [0.1, 0.9, 0.2, 0.8, 0.1]], dtype=tf.float32)
        alpha = 0.25
        gamma = 2.0
        class_weight_tensor = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32)

        # Calculate the loss using the custom focal loss function
        loss_value = focal_loss_with_class_weights(y_true, y_pred, alpha, gamma, class_weight_tensor)
        
        # Check if the loss value is a valid tensor and has the correct shape
        self.assertIsInstance(loss_value, tf.Tensor)
        self.assertEqual(loss_value.shape, ())

if __name__ == '__main__':
    unittest.main()
