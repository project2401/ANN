import csv
import pickle
import math
import numpy as np


if __name__ == '__main__':
    with open('data.pickle', 'rb') as f:
        data_new = pickle.load(f)


num1 = int(data_new[0][0])
num2 = int(data_new[0][1])
num3 = int(data_new[0][2])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict(num1, num2, num3):
    inputs = np.array([num1, num2, num3])
    weights_input_h1 = [0.1, 0.57, 0.44]
    weights_input_h2 = [0.8, 0.34, 0.34]
    weights_input_h3 = [0.88, 0.21, 0.37]
    weights_input_h4 = [0.44, 0.57, 0.39]
    weights_input_h = np.array([weights_input_h1, weights_input_h2, weights_input_h3, weights_input_h4])
    weights_h_to_o1 = np.array([0.96, 0.15, 0.35, 0.45])
    
    h_input = np.dot(weights_input_h, inputs)
    print("hiden_output = " +str(h_input))

    h_output = np.array([sigmoid(x) for x in h_input])
    print("h_output = "+str(h_output))

    o1_input = np.dot(weights_h_to_o1, h_output)
    print("o1_input = " + str(o1_input))
    o1_output = sigmoid(o1_input)
    return o1_output

print("o1_output = " + str(predict(num1, num2, num3)))



