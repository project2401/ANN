import csv
import pickle
import math
import numpy as np
import sys


if __name__ == '__main__':
    with open('data.pickle', 'rb') as f:
        data_new = pickle.load(f)

class Ann(object):
    def __init__(self, rate=0.1):
        # self.num1 = int(data_new[0][0])
        # self.num2 = int(data_new[0][1])
        # self.num3 = int(data_new[0][2])
        # self.num4 = int(data_new[0][3])
        self.learning_rate = np.array([rate])
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.weights_input_h1 = [0.1, 0.57, 0.44]
        self.weights_input_h2 = [0.8, 0.34, 0.34]
        self.weights_input_h3 = [0.88, 0.21, 0.37]
        self.weights_input_h4 = [0.44, 0.57, 0.39]
        self.weights_input_h = np.array([self.weights_input_h1, self.weights_input_h2, self.weights_input_h3, self.weights_input_h4])
        self.weights_h_to_o1 = np.array([0.96, 0.15, 0.35, 0.45])   

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        # inputs = np.array([num1, num2, num3])
        h_input = np.dot(self.weights_input_h, inputs)
        print("h_input = " +str(h_input))
        h_output = self.sigmoid_mapper(h_input)
        print("h_output = "+str(h_output))

        o1_input = np.dot(self.weights_h_to_o1, h_output)
        print("o1_input = " + str(o1_input))
        o1_output = self.sigmoid_mapper(o1_input)
        print('o1_output = ' + str(o1_output))
        return o1_output

    def train(self, inputs, num4):
        h_input = np.dot(self.weights_input_h, inputs)
        print("h_input = " +str(h_input))
        h_output = self.sigmoid_mapper(h_input)
        print("h_output = "+str(h_output))

        o1_input = np.dot(self.weights_h_to_o1, h_output)
        print("o1_input = " + str(o1_input))
        o1_output = self.sigmoid_mapper(o1_input)
        print('o1_output = ' + str(o1_output))
        actual_predict = o1_output

        error = np.array([actual_predict - num4]) #ошибка при вычислении 
        print('Error = '+str(error))
        sigmoidDX = actual_predict * (1 - actual_predict)
        weights_delta = error*sigmoidDX
        print('weights_delta = '+str(weights_delta))
        self.weights_input_h -= (np.dot(weights_delta, h_output.reshape(1, len(h_output)))) * self.learning_rate






# predict(num1, num2, num3, num4, rate)



