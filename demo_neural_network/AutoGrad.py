import math

class Value:
    def __init__(self, value, _parents=()):
        self.value=value
        self.grad=0
        self._backward=lambda: None
        self._prev=set(_parents)
    
    def __repr__(self): # prints the value and the gradient
        return f"V:{self.value} G:{self.grad}"
    
    def __add__(self,other): # A + B
        out = Value(self.value+other.value,(self,other))
        def _backward():
            self.grad+=out.grad * 1
            other.grad+=out.grad * 1
        out._backward=_backward
        return out
    
    def __mul__(self,other): # A * B
        out = Value(self.value*other.value,(self,other))
        def _backward():
            self.grad+=other.value*out.grad
            other.grad+=self.value*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self,value:int | float): # A ** {scalar}
        out = Value(self.value**value,(self,))
        def _backward():
            self.grad+=value*(self.value**(value-1))*out.grad
        out._backward=_backward
        return out
    
    def __truediv__(self,other): # A / B
        out = Value(self.value*(other.value**(-1)),(self,other))
        def _backward():
            self.grad+=other.value**(-1)*out.grad
            other.grad+=(-1)*(self.value*(other.value**(-2)))*out.grad
        out._backward=_backward
        return out

    def relu(self): # max(0,A)
        out = Value(0 if self.value<0 else self.value,(self,))
        def _backward():
            self.grad+=(out.value>0)*out.grad
        out._backward=_backward
        return out

    def topo_sort(self): # used to get the order for backpropagation
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self): # call backward functions in reverse order to get the gradients
        topo = self.topo_sort()
        self.grad=1
        for node in reversed(topo):
            node._backward()

    def zero_grads(self): # set all gradients to zero, useful for training loops
        topo = self.topo_sort()
        for node in topo:
            node.grad=0

    def __rpow__(self,value:int | float): # {scalar} ** A (used in e^x)
        out = Value(value**self.value,(self,))
        def _backward():
            self.grad+=value**self.value*out.grad*math.log(value)
        out._backward=_backward
        return out

    def log(self): # log(A)
        out = Value(math.log(self.value),(self,))
        def _backward():
            self.grad+=(1/self.value)*out.grad
        out._backward=_backward
        return out
