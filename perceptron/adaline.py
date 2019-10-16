import numpy as np


def activation(x):
    return 1 if x > 0 else -1

def adaline(input, output, weights, alpha = 0.02, max_error = 0.3, bipolar=True):
    result = np.ones(shape=output.shape)
    order = np.asarray(range(input.shape[0]))
    epochs = 0
    end = False
    while not end:
        end = True
        # print(weights)
        for i in order:
            result[i] = input[i] @ weights
            delta = output[i] - result[i]
            if(delta * delta > max_error):
                weights = weights + 2 * delta * alpha * input[i]
                end = False
        epochs += 1
    result = np.vectorize(activation)(result)
    return result, epochs