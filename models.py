import tensorflow.keras as keras

def get_simple_regression(num_features=102, multitask=False):
    out_dims = 2 if multitask else 1
    inputs = keras.layers.Input(shape=(num_features,))
    x = keras.layers.Dense(32)(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(8)(x)
    x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(out_dims)(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)
