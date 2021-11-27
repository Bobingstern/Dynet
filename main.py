from copy import deepcopy
from typing import List, Any
from random import random
from Dynet import Dynet, TANH, SIGMOID
import matplotlib.pyplot as plt  # pip install matplotlib
from numpy import tanh


PLAYER_COUNT = 100
TRAIN_GENERATIONS = 300
BRAIN_MUTATION_CHANCE = 0.2

# *** TEST CASES ***
# Uncomment one to try one out
# ---
# # OR test case
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [0,      1,      1,      1]
# ---
# # AND test case
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [0,      0,      0,      1]
# ---
# # NOT test case
# expectedInput = [[0], [1]]
# expectedOutput = [1,   0]
# ---
# # XOR test case
expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
expectedOutput = [0,      1,      1,      0]
# ---
# # Reverse XOR test case
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [1,      0,      0,      1]
# ---
# Greater or less
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [[0, 0], [1, 0], [0, 1], [0, 0]]
# ---
# 1 to 1
# expectedInput = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
# expectedOutput = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
# ---
# 1 to 0
# expectedInput = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
# expectedOutput = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
# ---


# # Greater than case
# expectedInput = []
# expectedOutput = []
# for _ in range(10):
#     a = random()
#     b = random()
#     expectedInput.append([a, b])
#     expectedOutput.append(int(a > b))
# ---
# # Array of 100 values is greater then 50 case
# expectedInput = []
# expectedOutput = []
# for _ in range(20):
#     values = [random() for _ in range(100)]
#     expectedInput.append(values)
#     expectedOutput.append(int(sum(values) > 50))
# ---
# # Array of 1000 values is greater then 500 case
# expectedInput = []
# expectedOutput = []
# for _ in range(50):
#     values = [random() for _ in range(1000)]
#     expectedInput.append(values)
#     expectedOutput.append(int(sum(values) > 500))
# ---
# # Addition
# expectedInput = []
# expectedOutput = []
# for _ in range(20):
#     values = [random() for _ in range(2)]
#     expectedInput.append(values)
#     expectedOutput.append(tanh(sum(values)))
# ---
# *** END TEST CASES ***

INPUTS = len(expectedInput[0])
try:
    OUTPUTS = len(expectedOutput[0])
except TypeError:
    OUTPUTS = 1


class Player:
    """
    Represents a simple player
    """
    def __init__(self):
        """
        Create a player
        """
        self.brain = Dynet(INPUTS, OUTPUTS, 0, TANH)
        self.brain.mutate(1, 10)
        self.fitness = 0


    def evaluate(self, allIns: List[Any], allOuts: List[int], log = False):
        """
        Evaluate the player

        :param allIns: A list of expected inputs
        :param allOuts: A list of expected outputs
        :param log: Whether to print what we were to expect
        """
        for index, input in enumerate(allIns):
            nnOut = self.brain.feedForward(input)
            if len(nnOut) == 1:
                self.fitness -= (nnOut[0] - allOuts[index]) ** 2
            else:
                totalError = 0
                for i, exp in enumerate(allOuts[index]):
                    totalError += (nnOut[i]-exp) ** 2
                self.fitness -= totalError
            if log:
                print(f"{input}\t"
                      f"Expected: {allOuts[index]}\t"
                      f"Actual: {nnOut}")

    def copy(self):
        return deepcopy(self)


def main():
    print("Hyper Dynamic Neural Network Concept")
    print("Created by Anik Patel")

    plt.xlabel("Generation")
    plt.ylabel("Error")
    plt.title("Average Error Over Generations")

    error = []
    avgErrors = []
    worstErrors = []
    gens = []

    players = []
    for i in range(PLAYER_COUNT):
        players.append(Player())

    bestEverFitness = -999
    bestEverPlayer = 0
    for g in range(TRAIN_GENERATIONS):
        for player in players:
            player.evaluate(expectedInput, expectedOutput)

        highest = -9999
        bestIndex = -1
        beatPrevious = False
        avgError = 0
        worstError = 0
        for i, player in enumerate(players):
            avgError += player.fitness
            if player.fitness > highest:
                highest = player.fitness
                bestIndex = i
            if player.fitness < worstError:
                worstError = player.fitness
            if player.fitness > bestEverFitness:
                bestEverFitness = player.fitness
                bestEverPlayer = player.copy()
                beatPrevious = True

        avgError /= len(players)
        avgErrors.append(-avgError)
        error.append(-highest)
        worstErrors.append(-worstError)
        gens.append(g)

        for player in players:
            player.fitness = 0
            player.brain = players[bestIndex].brain.copy()
            player.brain.mutate(BRAIN_MUTATION_CHANCE, 1)
        players[0].brain = bestEverPlayer.brain.copy()

        if beatPrevious:
            print(f"* Generation {g} best fitness: {highest}")
        else:
            print(f"Generation {g} best fitness: {highest}")

    bestEverPlayer.evaluate(expectedInput, expectedOutput, True)
    bestEverPlayer.brain.printNetwork()

    plt.plot(gens, error, label="Best of each gen", color="green")
    # plt.plot(gens, avgErrors, label="Average of each gen", color="blue")
    # plt.plot(gens, worstErrors, label="Worst of each gen", color="red")
    plt.legend()
    plt.grid()
    plt.show()
    print(error)

if __name__ == "__main__":
    main()
