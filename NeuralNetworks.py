# Features of training data should be fed in following format:
#1) X as a 2-Dimensional numpy.ndarray
#4) Y as a 1-Dimensional numpy.ndarray

import numpy as np

def sig(z):
    return (1 / (1 + np.exp(-z)))

def derivative_of_sig(z):
    return sig(z) * (1 - sig(z))

class Neuron:
    def __init__(self, num_of_inputs):
        self.weights = np.array(2 * np.random.random(num_of_inputs) - 1)
        self.bias = 2 * np.random.random() - 1
        self.weight_derivatives = np.zeros(num_of_inputs, dtype=float)
        self.bias_derivative = 0.0

    def update(self, lr):
        self.weights -= (self.weight_derivatives * lr)
        self.bias -= (self.bias_derivative * lr)
        self.weight_derivatives[:] = 0.0
        self.bias_derivative = 0.0

class NeuronLayer:
    def __init__(self):
        self.neurons = []
        self.output_values = []

    def add(self, neuron):
        self.neurons.append(neuron)
        self.output_values.append(0.0)

    def feed(self, input:np.ndarray):
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            z = np.dot(input.T, neuron.weights) + neuron.bias
            self.output_values[i] = sig(z)
        return self.output_values.copy()

    def print_weights_and_biases(self):
        print("number of neurons in output layer:", len(self.neurons))
        for i in range(len(self.neurons)):
            print(i)
            neuron = self.neurons[i]
            for weight_index in range(len(neuron.weights)):
                print("Weight of Neuron", i, "Weight", weight_index, "is", neuron.weights[weight_index])
            print("Bias of Neuron", i, "is", neuron.bias, "\n")

class NeuralNetwork:
    def __init__(self, hidden_layer_tuple = (), lr = 0.001, num_itr = 1000):
        self.lr = lr
        self.num_itr = num_itr
        self.hidden_layer_tuple = hidden_layer_tuple

    def predict(self, X):
        y_pred_encoded = self.predict_encoded(X)
        y_pred = []
        for i in range(len(y_pred_encoded)):
            y_pred.append(self.classes[np.argmax(np.array(y_pred_encoded[i]))])
        return y_pred

    def predict_encoded(self, X):
        y_pred_encoded = list()
        m = X.shape[0]
        for i in range(m):
            inp =self.predict_one_encoded(X[i])
            y_pred_encoded.append(inp)
        return y_pred_encoded

    def predict_one_encoded(self, x):
        inputs = x
        n = len(self.hidden_layers)
        if(n > 0):
            for i in range(n):
                inputs = np.array(self.hidden_layers[i].feed(inputs))
        output = self.output_layer.feed(inputs)
        return output

    def fit(self, X, Y):
        self.classes = list(set(Y))
        self.create_neural_network(self.hidden_layer_tuple, X)
        self.gd(X, Y, self.lr, self.num_itr)

    def create_neural_network(self, hidden_layer_tuple, X):
        n = len(hidden_layer_tuple)
        num_inputs = X.shape[1]
        self.hidden_layers = []
        if(n > 0):
            for i in range(n):
                hidden_layer = NeuronLayer()
                for curr_neuron in range(hidden_layer_tuple[i]):
                    neuron = Neuron(num_inputs)
                    hidden_layer.add(neuron)
                self.hidden_layers.append(hidden_layer)
                num_inputs = hidden_layer_tuple[i]
        self.output_layer = NeuronLayer()
        num_classes = len(self.classes)
        for i in range(num_classes):
            neuron = Neuron(num_inputs)
            self.output_layer.add(neuron)

    def gd(self, X, Y, lr, num_itr):
        ya = np.array(self.encode(Y))
        for i in range(num_itr):
            self.step_gradient(X, ya, lr)
            print(i, "Cost:" ,self.cost(ya, np.array(self.predict_encoded(X))))

    def step_gradient(self, X, ya, lr):
        hidden_layers = self.hidden_layers
        n = len(hidden_layers)
        m = X.shape[0]
        for i in range(m):
            y_pred = self.predict_one_encoded(X[i])
            next_list = []
            inputs = X[i]
            if(n > 0):
                inputs = np.array(hidden_layers[-1].output_values)
            for neuron_index in range(len(self.output_layer.neurons)):
                neuron = self.output_layer.neurons[neuron_index]
                y_p = y_pred[neuron_index]
                y_a = ya[i][neuron_index]
                first_term = ((-y_a)/(1 - y_a + y_p)) + ((1 - y_a)/(1 - y_p + y_a))
                inp = np.dot(inputs, neuron.weights) + neuron.bias
                second_term = derivative_of_sig(inp)
                for weight_index in range(len(neuron.weights)):
                    third_term = inputs[weight_index]
                    neuron.weight_derivatives[weight_index] += (first_term * second_term * third_term)
                neuron.bias_derivative += (first_term * second_term)
                next_list.append((first_term, second_term))
        
            if(n > 0):
                for layer_index in range(len(hidden_layers) - 1, -1, -1):
                    new_list = []
                    curr_layer = hidden_layers[layer_index]
                    inputs = X[i]
                    if(layer_index > 0):
                        inputs = np.array(hidden_layers[layer_index - 1].output_values)
                    for neuron_index in range(len(curr_layer.neurons)):
                        neuron = curr_layer.neurons[neuron_index]
                        first_term = 0.0
                        next_layer = self.output_layer
                        if(layer_index < n - 1):
                            next_layer = hidden_layers[layer_index + 1]
                        for next_layer_neuron_index in range(len(next_list)):
                            resp_wt = next_layer.neurons[next_layer_neuron_index].weights[neuron_index]
                            first_term += (next_list[next_layer_neuron_index][0] * next_list[next_layer_neuron_index][1] * resp_wt)
                        inp = np.dot(inputs, neuron.weights) + neuron.bias
                        second_term = derivative_of_sig(inp)
                        for weight_index in range(len(neuron.weights)):
                            third_term = inputs[weight_index]
                            neuron.weight_derivatives[weight_index] += (first_term * second_term * third_term)
                        neuron.bias_derivative += (first_term * second_term)
                        new_list.append((first_term, second_term))
                    next_list.clear()
                    next_list = new_list.copy()           
        if(n > 0):
            for i in range(n):
                curr_layer = hidden_layers[i]
                for neuron in curr_layer.neurons:
                    neuron.update(lr)
        for i in range(len(self.output_layer.neurons)):
            neuron = self.output_layer.neurons[i]
            neuron.update(lr)

    def cost(self, ya, yp):
        error = 0.0
        for i in range(ya.shape[0]):
            for j in range(ya.shape[1]):
                error += ((yp[i, j] - ya[i, j]) ** 2)/2
        return error

    def print_len_of_layers(self):
        n = len(self.hidden_layers)
        if n > 0:
            for i in range(n):
                print("Number of neurons in Hidden Layer", i, "are", len(self.hidden_layers[i].neurons), "and number of there inputs are", len(self.hidden_layers[i].neurons[0].weights))
        print("Number of neurons in output layer are", len(self.output_layer.neurons), "and number of there inputs are", len(self.output_layer.neurons[0].weights))

    def encode(self, Y):
        Y_encoded = np.zeros((len(Y), len(self.output_layer.output_values)), dtype=float)
        for i in range(len(Y)):
            y = Y[i]
            Y_encoded[i, self.classes.index(y)] = 1
        return Y_encoded