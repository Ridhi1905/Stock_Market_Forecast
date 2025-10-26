# src/utils.py
import random
import numpy as np
import tensorflow as tf

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
