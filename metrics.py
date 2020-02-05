import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.losses import poisson

def tih_poisson(y_true, y_pred):
    return poisson(y_true[:,0], y_pred[:,0])

def get_mask_loss(mask_value, loss_fn):
    def mask_loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        mask = K.equal(y_true, mask_value)
        mask = 1 - K.cast(mask, K.floatx())

        loss = loss_fn(y_true, y_pred)
        return loss * tf.cast(tf.size(y_true), K.floatx()) / K.sum(mask)
    return mask_loss

def convergence_point(loss_hist, conv_seq_len=20, ratio=0.5):
    for e in range(0, len(loss_hist) - conv_seq_len):
        loss = loss_hist[e]
        e_lt = 0
        for fut_e in range(e + 1, e + conv_seq_len):
            if loss_hist[fut_e] < loss:
                e_lt = e_lt + 1
        if conv_seq_len - e_lt >= conv_seq_len * ratio:
            return e
    return -1
