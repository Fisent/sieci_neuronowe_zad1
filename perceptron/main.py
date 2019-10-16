import numpy as np
from plots import *
from constants import *
from adaline import adaline
from perceptron import perceptron
from time import sleep






# def adaline(input, output, weights, alpha = 0.01, max_err=0.1):
#     result = np.ones(shape=output.shape)
#     is_finished = False
#     arr = np.asarray([0, 1, 2, 3])
#     while not is_finished:
#         np.random.shuffle(arr)
#         is_finished = True
#         for i in range(input.shape[0]):
#             result[i] = input[i] @ weights
#             delta = output[i] - result[i]
#             print(delta * delta)
#             if(delta * delta > max_err):
#                 is_finished = False
#                 weights = weights + 2 * delta * alpha * input[i]
#         print('                      weights:', weights)
#     return result

# def adaline(input, output, weights, alpha = 0.01, max_err=0.3):
#     result = np.ones(shape=output.shape)
#     is_finished = False
#     arr = np.asarray([0, 1, 2, 3])
#     while not is_finished:
#         np.random.shuffle(arr)
#         is_finished = True
#         for i in range(input.shape[0]):
#             result[i] = input[i] @ weights
#             delta = output[i] - result[i]
#             print(delta * delta)
#             if(delta * delta > max_err):
#                 is_finished = False
#                 weights = weights + 2 * delta * alpha * input[i]
#         print('                      weights:', weights)
#     return result


if __name__ == "__main__":
    # print('PERCEPTRON binary activation')
    # weights = np.random.rand(3)
    # print('AND')
    # print(perceptron(input=INPUT, output=AND_OUTPUT, weights=weights, bipolar=False))
    #
    # weights = np.random.rand(3)
    # print('OR')
    # print(perceptron(input=INPUT, output=OR_OUTPUT, weights=weights, bipolar=False))
    # #
    print('\n\nPERCEPTRON bipolar activation')
    weights = 2 * np.random.rand(3) -1
    print('AND')
    print(perceptron(input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR, weights=weights, bipolar=True))
    # #
    # print('\n\nADALINE')
    # weights = 2 * np.random.rand(3) - 1
    # print('AND')
    # print(adaline(input=INPUT_ADELINE, output=AND_OUTPUT_ADALINE, weights=weights))
    #
    # weights = 2 * np.random.rand(3) - 1
    # print('OR')
    # print(adaline(input=INPUT_ADELINE, output=OR_OUTPUT_ADALINE, weights=weights))

    plot_all()
