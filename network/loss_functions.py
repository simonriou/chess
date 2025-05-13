import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def loss_fn(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    
    cosine_sim = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    cosine_loss = 1.0 + cosine_sim  # [0, 2]

    huber = tf.keras.losses.huber(y_true, y_pred, delta=0.5)
    
    return 0.5 * huber + 0.5 * cosine_loss