import numpy as np

# def binary_activation(data):
#     return 1 if data > 0 else 0
#
#
# def bipolar_activation(data):
#     return 1 if data > 0 else -1
#
#
# def perceptron(input, output, weights, alpha = 0.01, bipolar = False):
#     result = np.ones(shape=output.shape)
#     end = False
#     epochs = 0
#     activation_function = bipolar_activation if bipolar else binary_activation
#     order = np.arange(input.shape[0])
#     while not end:
#         end = True
#         # print(result)
#         np.random.shuffle(order)
#         for i in order:
#             result[i] = input[i] @ weights
#             result[i] = np.vectorize(activation_function)(result[i])
#             delta = output[i] - result[i]
#             if delta is not 0:
#                 end = False
#                 weights = weights + delta * alpha * input[i]
#         epochs += 1
#     return result, epochs


def binary_activation(data, treshold):
    return 1 if data > treshold else 0


def bipolar_activation(data, treshold):
    return 1 if data > treshold else -1


def perceptron(input, output, weights, treshold=0.5, alpha=0.02, bipolar=True):
    result = np.ones(shape=output.shape)
    running = True
    epochs = 0
    activation_function = bipolar_activation if bipolar else binary_activation
    while running:
        running = False
        # print(weights)
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            result[i] = np.vectorize(activation_function)(result[i], treshold)
            delta = output[i] - result[i]
            if not delta == 0 :
                running = True
                weights = weights + delta * alpha * input[i]
        epochs += 1
    return result, epochs
