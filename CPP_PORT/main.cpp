#include <iostream>
#include <random>
#include <math.h>
#include <vector>
#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <future>
#include <vector>
#include <unistd.h>
#include<windows.h>
#include <iostream>
#include <fstream>
#include "Dynet.hpp"

using namespace std;

const int PLAYER_COUNT = 100;
const int TRAIN_GENERATIONS = 300;
const float BRAIN_MUTATION_CHANCE = 0.3;

//XOR Test case
vector<vector<float>> expectedInput = {{1, 0}, {1, 1}, {0, 0}, {0, 1}};
vector<vector<float>> expectedOutput = {{1}, {0}, {0}, {1}};

void printArr(vector<float> arr){
    for (int x=0;x<arr.size();x++){
        printf("%f ", arr[x]);
    }
}

class Player{
    public:
        Dynet brain = Dynet(2, 1, 1, TANH_);
        float fitness = 0;
    Player(){
        brain.mutate(1, 10);
    }
    void evaluate(bool LOG=false){
        fitness = 0;
        for (int i=0;i<expectedInput.size();i++){
            vector<float> out = brain.feedForward(expectedInput[i]);
            float totalError = 0;
            for (int j=0;j<out.size();j++){
                totalError += pow(out[j]-expectedOutput[i][j], 2);
            }
            if (LOG){
                printf("Input:[");
                printArr(expectedInput[i]);
                printf("] Expected: ");
                printArr(expectedOutput[i]);
                printf(" Actual: ");
                printArr(out);
                printf("\n");
            }
            fitness-=totalError;
        }
    }
};


int main(){
    srand(time(nullptr));
//    ofstream graphFile("graph.txt");
//    graphFile.open("graph.txt");

    vector<Player> players;
    for (int i=0;i<PLAYER_COUNT;i++){
        players.push_back(Player());
    }
    Player bestEver;
    float bestEverFit = -999;
    for (int g=0;g<TRAIN_GENERATIONS;g++){
        for (int p=0;p<players.size();p++){
            players[p].evaluate();
        }
        float highest = -999;
        int bestIndex = 0;
        for (int p=0;p<players.size();p++){
            if (players[p].fitness > highest){
                highest = players[p].fitness;
                bestIndex = p;
                if (highest > bestEverFit){
                    bestEverFit = highest;
                    bestEver.brain = players[p].brain.copy();
                }
            }
        }
        for (int p=0;p<players.size();p++){
            players[p].brain = players[bestIndex].brain.copy();
            players[p].brain.mutate(BRAIN_MUTATION_CHANCE, 1);
        }
        //players[0].brain = bestEver.brain;
        cout<<"Generation: "<<g<<" Highest fitness: "<<highest<<"\n";
        //graphFile << highest << "\n";
    }
    bestEver.evaluate(true);
    bestEver.brain.printNetwork();
    
    return 0;
}