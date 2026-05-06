import math


class Value:
    def __init__(self, data, _childs=()):
        self.grad = 0.0
        self.data = data
        self._back = lambda: None
        self._childs = set(_childs)

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _back():
            self.grad += out.grad
            other.grad += out.grad

        out._back = _back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._back = _back
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,))

        def _back():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._back = _back
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _back():
            self.grad += out.data * out.grad

        out._back = _back
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,))

        def _back():
            self.grad += (out.data > 0) * out.grad

        out._back = _back
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,))

        def _back():
            self.grad += (1.0 - out.data**2) * out.grad

        out._back = _back
        return out

    def backwards(self):
        topo = list()
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for ni in node._childs:
                    build_topo(ni)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._back()

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __rsub__(self, other):
        return (-self) + other
