from __future__ import annotations

from copy import deepcopy
from random import randint, choice, random, uniform as randfloat
from typing import List, Callable

from numpy import tanh
from numpy import exp

IN = 0
HIDDEN = 1
OUT = 2

SIGMOID = 0
TANH = 1

def sigmoid(x):
    return 1/(1 + exp(-x))


class Connection:
    """
    Represents a connection between two neurons
    """
    def __init__(self, layerFrom: int, indexFrom: int, weight: float):
        """
        Creates a connection.

        :param layerFrom: What layer (inout, hidden or output) either 0, 1, 2
        :param indexFrom: Index of the neuron in the layer
        :param weight: Weight of connection
        """
        self.layerFrom = layerFrom
        self.indexFrom = indexFrom
        self.weight = weight


class Neuron:
    """
    Represents a single Neuron
    """
    def __init__(self):
        """
        Creates a single neuron
        """
        self.connections = []
        self.outGoing = 0
        self.value = 0

    def addConnection(self, layerFrom: int, indexFrom: int, weight: float):
        """
        Add a connection.

        :param layerFrom: What layer (inout, hidden or output) either 0, 1, 2
        :param indexFrom: Index of the neuron in the layer
        :param weight: Weight of connection
        """
        self.connections.append(Connection(layerFrom, indexFrom, weight))


class Dynet:
    """
    Represents an entire Dynet
    """
    def __init__(self, inputs, outputs, hiddens=1, activation=SIGMOID):
        """
        Create an entire Dynet

        :param inputs: The number of input neurons
        :param hiddens: The number of hidden neurons
        :param outputs: The number of output neurons
        """
        h = hiddens
        if h <= 0:
            h = 1
        self.inputs = [Neuron() for _ in range(inputs+1)]
        self.hiddens = [Neuron() for _ in range(h)]
        self.outputs = [Neuron() for _ in range(outputs)]
        self.weightRange = 1
        self.activation = activation

    def activate(self, x):
        if self.activation == SIGMOID:
            return sigmoid(x)
        else:
            return tanh(x)

    def addRandomInputToHiddenConnection(self):
        """
        Adds a random input to the hidden layer
        """
        randomInputIndex = randint(0, len(self.inputs) - 1)
        randomHiddenIndex = randint(0, len(self.hiddens) - 1)
        self.hiddens[randomHiddenIndex].addConnection(
            IN, randomInputIndex, randfloat(-self.weightRange, self.weightRange)
        )
        self.inputs[randomInputIndex].outGoing += 1

    def addRandomInputToOutputConnection(self):
        """
        Adds a random output to the hidden layer
        """
        randomInputIndex = randint(0, len(self.inputs) - 1)
        randomOutputIndex = randint(0, len(self.outputs) - 1)
        self.outputs[randomOutputIndex].addConnection(
            IN, randomInputIndex, randfloat(-self.weightRange, self.weightRange)
        )
        self.inputs[randomInputIndex].outGoing += 1

    def addRandomHiddenToHiddenConnection(self):
        """
        Adds a random hidden to the hidden layer
        """
        randomHiddenIndex = randint(0, len(self.hiddens) - 1)
        randomHiddenIndex2 = randint(0, len(self.hiddens) - 1)
        self.hiddens[randomHiddenIndex2].addConnection(
            HIDDEN, randomHiddenIndex, randfloat(-self.weightRange, self.weightRange)
        )
        self.hiddens[randomHiddenIndex].outGoing += 1

    def addRandomHiddenToOutputConnection(self):
        """
        Adds a random hidden to the output layer
        """
        randomHiddenIndex = randint(0, len(self.hiddens) - 1)
        randomOutputIndex = randint(0, len(self.outputs) - 1)
        self.outputs[randomOutputIndex].addConnection(
            HIDDEN, randomHiddenIndex, randfloat(-self.weightRange, self.weightRange)
        )
        self.hiddens[randomHiddenIndex].outGoing += 1

    def addRandomConnection(self):
        """
        Add a random connection somewhere
        """
        funcs = [self.addRandomInputToHiddenConnection,
                 self.addRandomInputToOutputConnection,
                 self.addRandomHiddenToHiddenConnection,
                 self.addRandomHiddenToOutputConnection]
        try:
            choice(funcs)()
        except IndexError:
            pass

    def weightedSumHiddens(self):
        """
        Calculates the weighted sum of each hidden neuron and does stuff
        """
        for neuron in self.hiddens:
            neuron.value = 0
            if neuron.outGoing == 0:
                break
            for conn in neuron.connections:
                if conn.layerFrom == IN:
                    neuron.value += conn.weight * self.inputs[conn.indexFrom].value
                elif conn.layerFrom == HIDDEN:
                    neuron.value += conn.weight * self.hiddens[conn.indexFrom].value
            neuron.value = self.activate(neuron.value)

    def weightedSumOutputs(self):
        """
        Calculates the weighted sum of each output neuron and does stuff
        """
        for neuron in self.outputs:
            neuron.value = 0
            for conn in neuron.connections:
                if conn.layerFrom == HIDDEN:
                    neuron.value += conn.weight * self.hiddens[conn.indexFrom].value
                elif conn.layerFrom == IN:
                    neuron.value += conn.weight * self.inputs[conn.indexFrom].value
            neuron.value = self.activate(neuron.value)

    def mutate(self, rate: float, repeats: int, modifyHiddens=True):
        """
        Mutate stuff

        :param rate: Mutation rate
        :param repeats: How many times to repeat it
        """
        for _ in range(repeats):
            if random() < rate:
                self.addRandomConnection()
            if random() < rate / 2 and modifyHiddens:
                self.hiddens.append(Neuron())
    def printNetwork(self, printFunc: Callable = print):
        """
        Print the network.

        :param printFunc: The print function to use. Defaults to the default
         print function.
        """
        printFunc("Network: ")
        totalCons = 0
        for i in self.inputs:
            printFunc(f"{i.value:.2f} ", end="")
            totalCons += len(i.connections)
        printFunc("")
        for i in self.hiddens:
            printFunc(f"{i.value:.2f} ", end="")
            totalCons += len(i.connections)
        printFunc("")
        for i in self.outputs:
            printFunc(f"{i.value:.2f} ", end="")
            totalCons += len(i.connections)
        printFunc("")
        printFunc(f"Connection count: {totalCons}")

    def feedForward(self, ins: List[int]) -> List[float]:
        """
        Feed forward something

        :param ins: A list of integers
        """
        outputs = []
        for index, i in enumerate(ins):
            self.inputs[index].value = i
        self.inputs[len(self.inputs) - 1].value = 1
        self.weightedSumHiddens()
        self.weightedSumOutputs()
        for i in self.outputs:
            outputs.append(i.value)
        return outputs

    def copy(self) -> Dynet:
        """
        Returns a copy of oneself, for mutating
        """
        return deepcopy(self)
