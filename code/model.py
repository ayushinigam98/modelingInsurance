import numpy as np

def predicted_value(w: list, x: list)->float:
    #x = [1, f1, f2, ..., fn, ..., t]
    #w = [w0, w1, w2, ...]
    #remove the target value
    return np.dot(w, x)


