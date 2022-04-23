import numpy as np
import random

class NeuralNetwork:
    
    neurons_hidden = 32

    def __init__(self) -> None:
        self.weights_hidden = [[random.uniform(0, 1) for _ in range(64)] for _ in range(self.neurons_hidden)]
        self.weights_output = [random.uniform(0, 1) for _ in range(self.neurons_hidden)]

    def set_weights(self, filename):
        self.weights_hidden = []
        self.weights_output = []
        with open(filename, 'r') as source:
            lines = source.readlines()
            for x in range(self.neurons_hidden):
                data = lines[x].split('|')
                assert len(data) == 64
                self.weights_hidden.append([float(f) for f in data])
            
            data = lines[-1].split('|')
            assert len(data) == self.neurons_hidden
            self.weights_output = [float(f) for f in data]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def train(self, data, lr):
        input_values = data[0]
        expected_result = data[1]

        # input to hidden
        after_hidden = [self.sigmoid(np.dot(input_values, weights)) for weights in self.weights_hidden]
        assert len(after_hidden) == self.neurons_hidden

        # at output
        sum_at_output = np.dot(after_hidden, self.weights_output)
        actual_result = self.sigmoid(sum_at_output)

        error = actual_result - expected_result
        weights_delta = error * self.der_sigmoid(actual_result)
        
        for x in range(self.neurons_hidden):
            self.weights_output[x] = self.weights_output[x] - after_hidden[x] * weights_delta * lr

        errors_hidden = [x * weights_delta for x in self.weights_output]
        weights_delta_hidden = [errors_hidden[i] * self.der_sigmoid(after_hidden[i]) for i in range(len(after_hidden))]
        for x in range(self.neurons_hidden):
            for y in range(64):
                self.weights_hidden[x][y] = self.weights_hidden[x][y] - input_values[y] * weights_delta_hidden[x] * lr

        return error    

    def predict(self, input_values):

        # input to hidden
        after_hidden = [self.sigmoid(np.dot(input_values, weights)) for weights in self.weights_hidden]
        assert len(after_hidden) == self.neurons_hidden

        return self.sigmoid(np.dot(after_hidden, self.weights_output))

    def output_weights(self, memory_file):
        with open(memory_file, 'w') as out:
            for x in range(self.neurons_hidden):
                for y in range(64):
                    out.write(str(self.weights_hidden[x][y]))
                    if y != 63:
                        out.write("|")
                out.write("\n")
            for x in range(self.neurons_hidden):
                out.write(str(self.weights_output[x]))
                if x != self.neurons_hidden - 1:
                    out.write("|")
