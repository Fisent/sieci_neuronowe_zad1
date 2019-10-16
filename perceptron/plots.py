from adaline import adaline
from perceptron import perceptron
import numpy as np
from constants import *
import matplotlib.pyplot as plt


def plot_for_different_ranges(func, input, output):
    results = []
    for i in RANGES:
        sum = 0.0
        for j in range(NUMBER_OF_REPEATS_FOR_AVERAGE):
            weights = np.random.uniform(-i, i, 3)
            _, epochs = func(input, output, weights)
            sum += epochs
        results.append(sum / NUMBER_OF_REPEATS_FOR_AVERAGE)
    return results

def plot_for_different_alphas(func, input, output, bipolar):
    results = []
    for alpha in ALPHAS:
        sum = 0.0
        for j in range(NUMBER_OF_REPEATS_FOR_AVERAGE):
            weights = np.random.uniform(-1, 1, 3) if bipolar else np.random.uniform(0, 1, 3)
            _, epochs = func(input, output, weights.copy(), alpha=alpha, bipolar=bipolar)
            sum += epochs
        results.append(sum / NUMBER_OF_REPEATS_FOR_AVERAGE)
    return results

def plot_unipolar_and_bipolar(func):
    results = []
    for bipolar in [False, True]:
        sum = 0.0
        for i in range(NUMBER_OF_REPEATS_FOR_AVERAGE):
            weights = np.random.uniform(-1, 1, 3) if bipolar else np.random.uniform(0, 1, 3)
            input = INPUT_BIPOLAR if bipolar else INPUT
            output = AND_OUTPUT_BIPOLAR if bipolar else AND_OUTPUT
            _, epochs = func(input, output, weights, bipolar=bipolar)
            sum += epochs
        results.append(sum / NUMBER_OF_REPEATS_FOR_AVERAGE)
    return results


def draw_ranges():
    perceptron_bipolar_result = plot_for_different_ranges(perceptron, INPUT_BIPOLAR, AND_OUTPUT_BIPOLAR)
    adeline_result = plot_for_different_ranges(adaline, INPUT_ADELINE, AND_OUTPUT_ADALINE)
    plt.xlabel('zakres generowanych wag')
    plt.ylabel('srednia liczba epok')
    plt.plot(RANGES, perceptron_bipolar_result, 'r')
    plt.plot(RANGES, adeline_result, 'g')
    plt.legend(['Perceptron bipolarny', 'Adaline'])
    plt.show()


def draw_alphas():
    perceptron_binary_result = plot_for_different_alphas(perceptron, INPUT, AND_OUTPUT, bipolar=False)
    perceptron_bipolar_result = plot_for_different_alphas(perceptron, INPUT_BIPOLAR, AND_OUTPUT_BIPOLAR, bipolar=True)
    adaline_result = plot_for_different_alphas(adaline, INPUT_ADELINE, AND_OUTPUT_ADALINE, bipolar=True)
    plt.xlabel('wspolczynnik uczenia alpha')
    plt.ylabel('srednia liczba epok')
    plt.plot(ALPHAS, perceptron_binary_result)
    plt.plot(ALPHAS, perceptron_bipolar_result)
    plt.plot(ALPHAS, adaline_result)
    plt.legend(['Perceptron binarny', 'Perceptron bipolarny', 'Adaline'])
    plt.show()


def draw_uni_bi():
    perceptron_results = plot_unipolar_and_bipolar(perceptron)
    print(perceptron_results)
    plt.bar(['Perceptron bipolarny', 'Perceptron binarny'], perceptron_results)
    plt.show()



def plot_all():
    print('ADALINE AND')
    # ranges_results = plot_for_different_ranges(adaline, INPUT_ADELINE, AND_OUTPUT_ADALINE)
    # print(ranges_results)
    # plt.plot(ranges_results)
    # alphas_results = plot_for_different_alphas(adaline, INPUT_ADELINE, AND_OUTPUT_ADALINE)
    # print(alphas_results)
    # plt.plot(alphas_results)
    # bi_unipolar_results = plot_unipolar_and_bipolar(perceptron)
    # print(bi_unipolar_results)
    # plt.plot(bi_unipolar_results)
    # plt.show()


    # draw_ranges()
    # draw_alphas()
    draw_uni_bi()

