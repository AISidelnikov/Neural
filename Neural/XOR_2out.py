import numpy as np

training_data = open("dataset/Data_train.csv", 'r') 
training_data_list = training_data.readlines()
training_data.close()

class neuron_Net:
    def __init__(self, input_num, neuron_num, output_num, learningrate):
        self.weights = (np.random.rand(neuron_num, input_num) + 0.0)
        self.weights_out = (np.random.rand(output_num, neuron_num) + 0.0)
        self.lr = learningrate
        pass

    def train(self, inputs_list, targets_list):
        inputs_x = np.array(inputs_list, ndmin=2).T 
        targets_Y = np.array(targets_list, ndmin=2).T 
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1 + np.exp(-x1))
        x2 = np.dot(self.weights_out, y1)
        y2 = 1/(1 + np.exp(-x2))
        E = -(targets_Y - y2)
        E_hidden = np.dot(self.weights_out.T, E)
        self.weights_out -= self.lr * np.dot((E * y2) * (1.0 - y2), np.transpose(y1))
        self.weights -= self.lr * np.dot((E_hidden * y1 * (1.0 - y1)), np.transpose(inputs_x))
        pass

    def query(self, inputs_list):
        inputs_x = np.array(inputs_list, ndmin=2).T
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1 + np.exp(-x1))
        x2 = np.dot(self.weights_out, y1)
        y2 = 1/(1 + np.exp(-x2))
        return y2
    
data_input = 2
data_neuron = 4
data_output = 2
learningrate = 0.2
n = neuron_Net(data_input, data_neuron, data_output, learningrate)
epochs = 80000

for e in range(epochs):
    for i in training_data_list:
        all_values = i.split(',') 
        inputs_x = np.asfarray(all_values[1:])
        targets_Y = int(all_values[0])
        targets_Y = np.zeros(data_output) + 0.01
        targets_Y[int(all_values[0])] = 0.99
        n.train(inputs_x, targets_Y)                      

print('Весовые коэффициенты:\n', n.weights)

for i in training_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    outputs = n.query(inputs_x)
    print(int(all_values[1]), 'XOR', int(all_values[2]), '=' ,  np.argmax(outputs), '\n',outputs)  