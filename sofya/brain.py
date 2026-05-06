import random
from sofya.heart import Value


class Perceptron:
    def __init__(self, nin):
        self.nin = nin
        self.bias = Value(random.uniform(-0.1, 0.1))
        self.weights = [Value(random.uniform(-0.5, 0.5)) for _ in range(nin)]

    def __call__(self, xin):
        assert len(xin) == self.nin
        return sum([wi * xi for wi, xi in zip(self.weights, xin)], self.bias)

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin, nout):
        self.perceptrons = [Perceptron(nin) for _ in range(nout)]

    def __call__(self, xin):
        return [perceptron(xin) for perceptron in self.perceptrons]

    def parameters(self):
        return [p for perceptron in self.perceptrons for p in perceptron.parameters()]


class MLP:
    def __init__(self, nin, nouts, activations):
        szs = [nin] + nouts
        self.layers = [Layer(szs[i], szs[i + 1]) for i in range(len(nouts))]
        self.activations = activations

    def __call__(self, xin):
        for layer, activation in zip(self.layers, self.activations):
            xin = layer(xin)

            if activation == "relu":
                xin = [x.relu() for x in xin]
            elif activation == "tanh":
                xin = [x.tanh() for x in xin]

        return xin

    def grad_zero(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
