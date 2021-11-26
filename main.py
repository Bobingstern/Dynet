from copy import deepcopy
from typing import List, Any

import matplotlib.pyplot as plt  # pip install matplotlib

from Dynet import Dynet

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
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [0,      1,      1,      0]
# ---
# # Reverse XOR test case
# expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
# expectedOutput = [1,      0,      0,      1]
# ---
# Greater or less
expectedInput = [[0, 0], [1, 0], [0, 1], [1, 1]]
expectedOutput = [0.5,    1,      0,      0.5]
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
        self.brain = Dynet(INPUTS, OUTPUTS, 1)
        self.fitness = 0
    def evaluate(self, log = False):
        """
        Evaluate the player

        :param log: Whether to print what we were to expect
        """
        allXorIns = [[0, 0], [1, 0], [0, 1], [1, 1]]
        correspondingOuts = [0, 1, 1, 0]
        for index, input in enumerate(allXorIns):
            nnOut = self.brain.feedForward(input)
            self.fitness -= (nnOut[0] - correspondingOuts[index]) ** 2
            if log:
                print(f"{input}\t"
                      f"Expected: {correspondingOuts[index]}\t"
                      f"Actual: {nnOut[0]}")

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
            player.evaluate()

        highest = -9999
        bestIndex = -1
        for i, player in enumerate(players):
            avgError += player.fitness
            if player.fitness > highest:
                highest = player.fitness
                bestIndex = i
            if player.fitness > bestEverFitness:
                bestEverFitness = player.fitness
                bestEverPlayer = player.copy()
                print(f"* New best fitness: {bestEverFitness}")

        for player in players:
            player.fitness = 0
            player.brain = players[bestIndex].brain.copy()
            player.brain.mutate(BRAIN_MUTATION_CHANCE, 1)

        print(f"Generation {g} best fitness: {highest}")

    bestEverPlayer.evaluate(expectedInput, expectedOutput, True)
    bestEverPlayer.brain.printNetwork()

    plt.plot(gens, error, label="Best of each gen", color="green")
    # plt.plot(gens, avgErrors, label="Average of each gen", color="blue")
    # plt.plot(gens, worstErrors, label="Worst of each gen", color="red")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
