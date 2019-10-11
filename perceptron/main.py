import numpy as np
from time import sleep


INPUT = np.asarray([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])


INPUT_BIPOLAR = np.asarray([[1, -1,-1], [1, -1,1], [1, 1,-1], [1, 1, 1]])
AND_OUTPUT_BIPOLAR = np.asarray([[-1], [-1], [-1], [1]])
OR_OUTPUT_BIPOLAR = np.asarray([[-1], [1], [1], [1]])


INPUT_ADELINE = np.asarray([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
AND_OUTPUT_ADALINE = AND_OUTPUT_BIPOLAR
OR_OUTPUT_ADALINE = OR_OUTPUT_BIPOLAR


def binary_activation(data, treshold):
    return 1 if data > treshold else 0


def bipolar_activation(data, treshold):
    return 1 if data > treshold else -1


def perceptron(input, output, weights, treshold = 0.5, alpha = 0.02, bipolar = False):
    result = np.ones(shape=output.shape)
    running = True
    activation_function = bipolar_activation if bipolar else binary_activation
    while running:
        running = False
        print(weights)
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            result[i] = np.vectorize(activation_function)(result[i], treshold)
            delta = output[i] - result[i]
            if(not delta == 0):
                running = True
                weights = weights + delta * alpha * input[i]
    return result


def adaline(input, output, weights, alpha = 0.02, max_error = 0.1):
    result = np.ones(shape=output.shape)
    running = True
    while running:
        print(weights)
        for i in range(input.shape[0]):
            result[i] = input[i] @ weights
            delta = output[i] - result[i]
            if(delta * delta > max_error):
                weights = weights + delta * alpha * input[i]
            else:
                running = False
    return weights


if __name__ == "__main__":
    print('PERCEPTRON binary activation')
    weights = np.random.rand(3)
    print('AND')
    print(perceptron(input=INPUT, output=AND_OUTPUT, weights=weights, bipolar=False))

    weights = np.random.rand(3)
    print('OR')
    print(perceptron(input=INPUT, output=OR_OUTPUT, weights=weights, bipolar=False))

    # print('\n\nPERCEPTRON bipolar activation')
    # weights = 2 * np.random.rand(3) -1
    # print('AND')
    # print(perceptron(input=INPUT_BIPOLAR, output=AND_OUTPUT_BIPOLAR, weights=weights, bipolar=True, treshold=0))

    print('\n\nADALINE')
    weights = 2 * np.random.rand(3) - 1
    print('AND')
    print(adaline(input=INPUT_ADELINE, output=AND_OUTPUT_ADALINE, weights=weights))

    weights = 2 * np.random.rand(3) - 1
    print('OR')
    print(adaline(input=INPUT_ADELINE, output=OR_OUTPUT_ADALINE, weights=weights))
