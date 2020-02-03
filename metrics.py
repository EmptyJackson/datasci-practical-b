from tensorflow.keras.losses import poisson

def tih_poisson(y_true, y_pred):
    return poisson(y_true[:,0], y_pred[:,0])
