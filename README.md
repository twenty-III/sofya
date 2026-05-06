# Sofya's Brain & Heart

A lightweight, purely vanilla Python autograd engine and neural network library built completely from scratch. 

No PyTorch. No TensorFlow. Not even NumPy. Just pure math, standard Python features, and a lot of chain rule calculus.

## 🧠 What is this?
This project is an educational deep learning framework. It implements a scalar-valued autograd engine that dynamically builds a Directed Acyclic Graph (DAG) of mathematical operations. It then uses a topological sort to perfectly calculate gradients and backpropagate them through the network.

While modern frameworks use highly optimized C++ tensor operations, this engine builds neural networks one scalar weight at a time. It's not designed to train massive LLMs, but it is a mathematically flawless demonstration of how deep learning actually works under the hood.

## ✨ Features
* **Custom Autograd Engine:** A `Value` object that tracks data and gradients for complex mathematical expressions.
* **Dynamic Computation Graphs:** Automatically builds the graph and calculates local derivatives on the fly.
* **Topological Sort Backprop:** Guarantees gradients flow backward in the exact correct order.
* **Standard Network Architecture:** Includes `Perceptron`, `Layer`, and `MLP` classes that mirror PyTorch's `nn.Linear` logic.
* **Activations & Loss:** Supports `ReLU`, `Tanh`, `Linear`, and Softmax activations, along with custom loss function implementations (MSE, Cross-Entropy).

## 🚀 The Sine Wave Sanity Check
To prove the engine's capability to learn non-linear continuous functions, the network was trained to approximate a sine wave. 

By utilizing **Tanh** activations to prevent the "Dying ReLU" problem, implementing learning rate decay, and ensuring an even distribution of dataset points, the network successfully learned the curves using a standard Gradient Descent optimization loop.

*(Consider adding one of the screenshots of your perfectly fitted sine wave curve right here!)*

## 🛠️ Why build this?
Because understanding the vanishing gradient problem, the math behind piece-wise linear approximations, and the sheer overhead of Python object creation is a rite of passage.
