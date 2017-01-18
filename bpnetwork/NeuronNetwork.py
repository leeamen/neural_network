import numpy as np
import basic


class Neuron:
    def __init__(self, len_input):
        self.weights = np.random.random(len_input) * 0.1
        self.input = np.ones(len_input)
        self.output = 1
        self.deltas_item = 0
        self.last_weight_add = 0

    def calc_output(self, x):
        self.input = x
        self.output = basic.sigmoid(np.dot(self.weights.T, self.input))
        return self.output

    def get_back_weight(self):
        return self.weights * self.deltas_item

    def update_weight(self, target=0, back_weight=0, learning_rate=0.1, layer='OUTPUT'):
        if layer == 'OUTPUT':
            self.deltas_item = (target - self.output) * basic.sigmoid_derivative(self.output)
        elif layer == 'HIDDEN':
            self.deltas_item = back_weight * basic.sigmoid_derivative(self.output)

        weight_add = self.input * self.deltas_item * learning_rate + 0.9 * self.last_weight_add
        self.weights += weight_add
        self.last_weight_add = weight_add


class NetLayer:
    def __init__(self, len_node, in_count):
        self.neurons = [Neuron(in_count) for _ in range(len_node)]
        self.next_layer = None

    def calc_output(self, x):
        output = np.array([node.calc_output(x) for node in self.neurons])
        if self.next_layer is not None:
            return self.next_layer.calc_output(output)
        return output

    def get_back_weight(self):
        return sum([node.get_back_weight() for node in self.neurons])

    def update_weight(self, learning_rate, target):
        layer = 'OUTPUT'
        back_weight = np.zeros(len(self.neurons))
        if self.next_layer is not None:
            back_weight = self.next_layer.update_weight(learning_rate, target)
            layer = 'HIDDEN'
        for i, node in enumerate(self.neurons):
            target_item = 0 if len(target) <= i else target[i]
            node.update_weight(target=target_item, back_weight=back_weight[i], learning_rate=learning_rate, layer=layer)
        return self.get_back_weight()


class NeuralNetWork:
    def __init__(self, layers):
        self.layers = []
        self.construct_networks(layers)
        pass

    def construct_networks(self, layers):
        last_layer = None
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            cur_layer = NetLayer(layer, layers[i - 1])
            self.layers.append(cur_layer)
            if last_layer is not None:
                last_layer.next_layer = cur_layer
            last_layer = cur_layer

    def fit(self, x_train, y_train, learning_rate=0.1, epochs=100000, shuffle=False):
        indices = np.arange(len(x_train))
        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                self.layers[0].calc_output(x_train[i])
                # self.layers[0].update_weight(learning_rate, y_train[i])
                self.layers[0].update_weight(learning_rate, np.array([y_train[i]]))
        pass

    def predict(self, x):
        return self.layers[0].calc_output(x)

