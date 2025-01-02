import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from graphviz import Digraph
from engine import Value
from neural_net import Neuron, Layer, MLP


def main():
    show_torch_version()
    test_neuron()

    # generate_graph()
    # generate_graph_using_math()
    generate_tensors()


def show_torch_version():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))


def generate_tensors():
    print("Generating tensors")

    x1 = torch.Tensor([2.0]).double()
    x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()
    x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()
    w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True
    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)

    print(o.data.item())
    o.backward()

    print("---")
    print("x2", x2.grad.item())
    print("w2", w2.grad.item())
    print("x1", x1.grad.item())
    print("w1", w1.grad.item())


def generate_graph():
    print("Generating graph")

    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735870195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"

    """
    o.grad = 1.0
    o._backward()
    print(o.grad)
    n._backward()
    x1w1x2w2._backward()
    x1w1._backward()
    x2w2._backward()
    """

    o.backward()
    draw_dot(o).render("graph1", format="png", cleanup=True)


def generate_graph_using_math():
    print("Generating graph using mathematic ecuation")

    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias of the neuron
    b = Value(6.8813735870195432, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    # ----
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    # ----
    o.label = "o"
    o.backward()

    draw_dot(o).render("graph2", format="png", cleanup=True)


def test_neuron():
    # n = MLP(2, [4, 4, 2])
    # x = [2.0, 0.0, -1.0]
    # y = n(x)
    # print(y)
    # draw_dot(y[0]).render("neuron_graph", format="png", cleanup=True)

    N_PREDS = 20

    n = MLP(3, [4, 4, 1])

    print(len(n.parameters()))

    # Input data
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    # Expected output data
    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(N_PREDS):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)
        print([y.data for y in ypred])

    draw_dot(loss).render("images/testing_neurons", format="png", cleanup=True)


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    main()
