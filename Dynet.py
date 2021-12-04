from __future__ import annotations

from copy import deepcopy
from random import randint, choice, random, uniform as randfloat
from typing import List, Callable

from numpy import tanh
from numpy import exp

import time

# //  _____                _           _   ______
# // /  __ \              | |         | |  | ___ \
# // | /  \/_ __ ___  __ _| |_ ___  __| |  | |_/ /_   _
# // | |   | '__/ _ \/ _` | __/ _ \/ _` |  | ___ \ | | |
# // | \__/\ | |  __/ (_| | ||  __/ (_| |  | |_/ / |_| |
# //  \____/_|  \___|\__,_|\__\___|\__,_|  \____/ \__, |
# //                                               __/ |
# //                                              |___/
# //   ___        _ _     ______     _       _
# //  / _ \      (_) |    | ___ \   | |     | |
# // / /_\ \_ __  _| | __ | |_/ /_ _| |_ ___| |
# // |  _  | '_ \| | |/ / |  __/ _` | __/ _ \ |
# // | | | | | | | |   <  | | | (_| | ||  __/ |
# // \_| |_/_| |_|_|_|\_\ \_|  \__,_|\__\___|_|


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
        self.outGoingConnections = []
        self.value = 0
        self.bias = 0

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
        self.inputs = [Neuron() for _ in range(inputs)]
        self.hiddens = [Neuron() for _ in range(h)]
        self.outputs = [Neuron() for _ in range(outputs)]
        self.weightRange = 1
        self.activation = activation

    def activate(self, x):
        if self.activation == SIGMOID:
            return sigmoid(x)
        else:
            return tanh(x)

    def derivative_activate(self, out):
        if self.activation == SIGMOID:
            return out * (1 - out)

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
        self.inputs[randomInputIndex].outGoingConnections.append([OUT, randomOutputIndex, len(self.outputs[randomOutputIndex].connections)-1])

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
        self.hiddens[randomHiddenIndex].outGoingConnections.append([HIDDEN, randomHiddenIndex2, len(self.hiddens[randomHiddenIndex2].connections)-1])

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
        self.hiddens[randomHiddenIndex].outGoingConnections.append([OUT, randomOutputIndex, len(self.outputs[randomOutputIndex].connections)-1])

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

    def removeRandomConnectionHiddens(self):

        randomHiddenIndex = randint(0, len(self.hiddens) - 1)
        if len(self.hiddens[randomHiddenIndex].connections) > 0:
            randomConnectionIndex = randint(0, len(self.hiddens[randomHiddenIndex].connections)-1)
            self.hiddens[randomHiddenIndex].connections.pop(randomConnectionIndex)
    def removeRandomConnectionOutputs(self):
        randomOutputIndex = randint(0, len(self.outputs) - 1)
        if len(self.outputs[randomOutputIndex].connections) > 0:
            randomConnectionIndex = randint(0, len(self.outputs[randomOutputIndex].connections)-1)
            self.outputs[randomOutputIndex].connections.pop(randomConnectionIndex)


    def removeRandomConnection(self):
        if random() < 0.5:
            self.removeRandomConnectionHiddens()
        else:
            self.removeRandomConnectionOutputs()

    def mutateRandomConnection(self):
        if random() < 0.5:
            randomHiddenIndex = randint(0, len(self.hiddens) - 1)
            if len(self.hiddens[randomHiddenIndex].connections) > 0:
                rConn = randint(0, len(self.hiddens[randomHiddenIndex].connections)-1)
                self.hiddens[randomHiddenIndex].connections[rConn].weight+=randfloat(-0.1, 0.1)
        else:
            randomOutputIndex = randint(0, len(self.outputs) - 1)
            if len(self.outputs[randomOutputIndex].connections) > 0:
                rConn = randint(0, len(self.outputs[randomOutputIndex].connections) - 1)
                self.outputs[randomOutputIndex].connections[rConn].weight += randfloat(-0.1, 0.1)

    def weightedSumHiddens(self):
        """
        Calculates the weighted sum of each hidden neuron and does stuff
        """
        for neuron in self.hiddens:
            neuron.value = 0
            if neuron.outGoing == 0:
                continue
            for conn in neuron.connections:
                if conn.layerFrom == IN:
                    neuron.value += conn.weight * self.inputs[conn.indexFrom].value
                elif conn.layerFrom == HIDDEN:
                    neuron.value += conn.weight * self.hiddens[conn.indexFrom].value
            neuron.value += neuron.bias
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


    def mutateBias(self):
        if random() < 0.5:
            r = randint(0, len(self.outputs)-1)
            self.outputs[r].bias += randfloat(-0.1, 0.1)
        elif len(self.hiddens) > 0:
            r = randint(0, len(self.hiddens) - 1)
            self.hiddens[r].bias += randfloat(-0.1, 0.1)

    def mutate(self, rate: float, repeats: int, modifyHiddens=True):
        """
        Mutate stuff

        :param rate: Mutation rate
        :param repeats: How many times to repeat it
        """
        for _ in range(repeats):
            if random() < rate / 2:
                self.addRandomConnection()
            if random() < rate:
                self.mutateRandomConnection()
            if random() < rate / 20 and modifyHiddens:
                self.hiddens.append(Neuron())
            if random() < rate / 2:
                self.removeRandomConnection()
            if random() < rate:
                self.mutateBias()

    def fullyConnect(self):
        #Connect all Inputs to all hiddens
        for index, _ in enumerate(self.inputs):
            for hidden in self.hiddens:
                hidden.addConnection(IN, index, randfloat(-self.weightRange, self.weightRange))
                self.inputs[index].outGoing += 1

        #Connect all hiddens to each other
        for index, _ in enumerate(self.hiddens):
            for i, hidden in enumerate(self.hiddens):
                if i == index:
                    continue
                hidden.addConnection(HIDDEN, index, randfloat(-self.weightRange, self.weightRange))
                self.hiddens[index].outGoing += 1

        #Connect all hiddens to outputs:
        for index, _ in enumerate(self.hiddens):
            for output in self.outputs:
                output.addConnection(HIDDEN, index, randfloat(-self.weightRange, self.weightRange))
                self.hiddens[index].outGoing+=1

    def backpropagate(self, expected):
        outputs = []
        outputErrors = []
        hiddenOutputError = []
        hiddenHiddenError = []
        inputHiddenError = []
        ##Calculate outputs errors
        for i in self.outputs:
            outputs.append(i.value)
        for index, i in enumerate(outputs):
            outputErrors.append((i-expected[index]) ** 2)

        ##Calculate hidden to output error
        for index, output in enumerate(self.outputs):
            totalError = 0
            for conIndex, conn in enumerate(output.connections):
                totalError += (conn.weight * outputErrors[index]) * self.derivative_activate(self.hiddens[conn.indexFrom].value)
            hiddenOutputError.append(totalError)

        # for index, hidden in enumerate(self.hiddens):
        #     totalError = 0
        #     for conIndex, conn in enumerate(hidden.connections):
        #         if (conn.layerFrom != HIDDEN):
        #             continue
        #         totalError += (conn.weight * hiddenOutputError[index]) * self.derivative_activate(self.hiddens[conn.indexFrom].value)
        #     hiddenHiddenError.append(totalError)

        print(outputErrors, hiddenOutputError)



    def feedForward(self, ins: List[int], printTime=False) -> List[float]:
        """
        Feed forward something

        :param ins: A list of integers
        """
        t = time.perf_counter()
        outputs = []
        for index, i in enumerate(ins):
            self.inputs[index].value = i
        self.weightedSumHiddens()
        self.weightedSumOutputs()
        for i in self.outputs:
            outputs.append(i.value)

        if printTime:
            print(time.perf_counter()-t)
        return outputs

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

    def copy(self) -> Dynet:
        """
        Returns a copy of oneself, for mutating
        """
        return deepcopy(self)
