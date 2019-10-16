import numpy as np

INPUT = np.asarray([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
AND_OUTPUT = np.asarray([[0], [0], [0], [1]])
OR_OUTPUT = np.asarray([[0], [1], [1], [1]])


INPUT_BIPOLAR = np.asarray([[1, -1,-1], [1, -1,1], [1, 1,-1], [1, 1, 1]])
AND_OUTPUT_BIPOLAR = np.asarray([[-1], [-1], [-1], [1]])
OR_OUTPUT_BIPOLAR = np.asarray([[-1], [1], [1], [1]])


INPUT_ADELINE = np.asarray([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
AND_OUTPUT_ADALINE = AND_OUTPUT_BIPOLAR
OR_OUTPUT_ADALINE = OR_OUTPUT_BIPOLAR

NUMBER_OF_REPEATS_FOR_AVERAGE = 100
RANGES = [0.2, 0.4, 0.6, 0.8, 1]
ALPHAS = [0.005, 0.01,  0.015, 0.02, 0.025, 0.03]

