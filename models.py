import tensorflow.keras as keras

def get_simple_regression(num_features=102, multitask=False):
    out_dims = 2 if multitask else 1
    inputs = keras.layers.Input(shape=(num_features,))
    x = keras.layers.Dense(32)(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(32)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(8)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(out_dims)(x)
    outputs = keras.layers.ReLU()(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)


def get_seq_model(num_features=102, seq_len=5, dropout=False):
    inputs = keras.layers.Input(shape=(seq_len, num_features))
    x = keras.layers.Masking(
        mask_value=0, input_shape=(seq_len, num_features))(inputs)
    x = keras.layers.Dense(32)(x)
    x = keras.layers.ReLU()(x)
    #x = keras.layers.Dense(32)(x)
    #x = keras.layers.ReLU()(x)
    if dropout:
        x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.GRU(units=32, return_sequences=True)(x)
    if dropout:
        x = keras.layers.Dropout(0.25)(x)
    #x = keras.layers.GRU(units=32, return_sequences=True)(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(8)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.ReLU()(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)
