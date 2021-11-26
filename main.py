from copy import deepcopy

from Dynet import Dynet

PLAYER_COUNT = 100
TRAIN_GENERATIONS = 150
BRAIN_MUTATION_CHANCE = 0.3

class Player:
    """
    Represents a simple player
    """
    def __init__(self):
        """
        Create a player
        """
        self.brain = Dynet(2, 1, 1)
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

    bestEverPlayer.evaluate(True)

if __name__ == "__main__":
    main()
