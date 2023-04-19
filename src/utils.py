import numpy as np

def discretize(y, pos_threshold, neg_threshold):
    y_dis = np.zeros_like(y)
    for i in range(y.shape[0]):
        if y[i] >= pos_threshold or y[i] <= neg_threshold:
            y_dis[i] = 1
        else:
            y_dis[i] = 0
    return y_dis