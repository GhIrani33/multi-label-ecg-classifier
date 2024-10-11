import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout,
                                     BatchNormalization, concatenate, SpatialDropout1D,
                                     Bidirectional, LSTM, Activation, Add)

# Define Focal Loss function with Class Weights
def focal_loss_with_class_weights(y_true, y_pred, alpha, gamma, class_weight_tensor):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    weights = class_weight_tensor  # Apply class weights
    focal_loss_value = weights * alpha * tf.pow((1 - bce_exp), gamma) * bce
    return tf.reduce_mean(focal_loss_value)

# Define Residual Block with shortcut adjustment
def residual_block(x, filters, kernel_size):
    shortcut = x
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    # Adjust shortcut
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    return x

# Define Attention Layer
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.tensordot(x, self.W, axes=1))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Define model creation function
def create_model(hparams, input_shape_ecg, input_shape_meta, num_classes):
    # ECG Input
    ecg_input = Input(shape=input_shape_ecg, name='ecg_input')
    x = ecg_input

    # Convolutional layers with residual connections
    for i in range(hparams['num_res_blocks']):
        x = residual_block(x, filters=hparams['filters'][i], kernel_size=hparams['kernel_sizes'][i])
        x = SpatialDropout1D(rate=hparams['dropout_rate'])(x)

    # Recurrent layer
    x = Bidirectional(LSTM(hparams['lstm_units'], return_sequences=True))(x)

    # Attention layer
    x = Attention()(x)

    x = Dense(hparams['dense_units'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=hparams['dropout_rate'])(x)

    # Metadata Input
    meta_input = Input(shape=input_shape_meta, name='meta_input')
    meta_x = Dense(hparams['meta_dense_units'], activation='relu')(meta_input)
    meta_x = BatchNormalization()(meta_x)
    meta_x = Dropout(rate=hparams['dropout_rate'])(meta_x)

    # Combine Inputs
    combined = concatenate([x, meta_x])
    combined = Dense(hparams['combined_dense_units'], activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(rate=hparams['dropout_rate'])(combined)
    outputs = Dense(num_classes, activation='sigmoid')(combined)

    model = Model(inputs=[ecg_input, meta_input], outputs=outputs)
    return model