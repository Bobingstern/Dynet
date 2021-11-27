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

const float E = 2.7182818284590452353602874;

int IN_ = 0;
int HIDDEN_ = 1;
int OUT_ = 2;

int SIGMOID_ = 0;
int TANH_ = 1;

using namespace std;

float sigmoid(float x){
    return 1/(1+exp(-x));
}

float randf(float lo, float hi) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = hi - lo;
    float r = random * diff;
    return lo + r;
}

float randint(int lo, int hi){
    return rand() % (hi+1) + lo;
}


class Connection{
    public:
        int layerFrom;
        int indexFrom;
        float weight;
    Connection(int lf, int If, float w){
        layerFrom = lf;
        indexFrom = If;
        weight = w;
    }  
  
};

class Neuron{
    public:
         vector<Connection> connections;
         int outGoing = 0;
         float value = 0;
         float bias = 0;
    Neuron(){

    }
    void addConnection(int lf, int If, float weight){
        connections.push_back(Connection(lf, If, weight));
    }
};

class Dynet{
    public:
        vector<Neuron> inputs;
        vector<Neuron> hiddens;
        vector<Neuron> outputs;
        float weightRange = 1;
        int activation;
    Dynet(int ins, int outs, int hids, int a=SIGMOID_){
        int h=1;
        if (hids > h){
            h = hids;
        }
        for (int i=0;i<ins;i++){
            inputs.push_back(Neuron());
        }
        for (int i=0;i<outs;i++){
            outputs.push_back(Neuron());
        }
        for (int i=0;i<h;i++){
            hiddens.push_back(Neuron());
        }
        activation = a;
    }
    
    float activate(float x){
        if (activation == SIGMOID_){
            return sigmoid(x);
        }
        else{
            return tanh(x);
        }
    }
    
    void addRandomInputToHiddenConnection(){
        int randomInputIndex = randint(0, inputs.size() - 1);
        int randomHiddenIndex = randint(0, hiddens.size() - 1);
        
        hiddens[randomHiddenIndex].addConnection(
            IN_, randomInputIndex, randf(-weightRange, weightRange)
        );
        inputs[randomInputIndex].outGoing += 1;
    }
    void addRandomInputToOutputConnection(){
        int randomInputIndex = randint(0, inputs.size() - 1);
        int randomOutputIndex = randint(0, outputs.size() - 1);
        outputs[randomOutputIndex].addConnection(
            IN_, randomInputIndex, randf(-weightRange, weightRange)
        );
        inputs[randomInputIndex].outGoing += 1;
    }
    void addRandomHiddenToOutputConnection(){
        int randomHiddenIndex = randint(0, hiddens.size() - 1);
        int randomOutputIndex = randint(0, outputs.size() - 1);
        
        outputs[randomOutputIndex].addConnection(
            HIDDEN_, randomHiddenIndex, randf(-weightRange, weightRange)
        );
        hiddens[randomHiddenIndex].outGoing += 1;
    }
    void addRandomHiddenToHiddenConnection(){
        int randomHiddenIndex = randint(0, hiddens.size() - 1);
        int randomHiddenIndex2 = randint(0, hiddens.size() - 1);
        hiddens[randomHiddenIndex2].addConnection(
            HIDDEN_, randomHiddenIndex, randf(-weightRange, weightRange)
        );
        hiddens[randomHiddenIndex].outGoing += 1;
    }
    void addRandomConnection(){
        int r = randint(0, 4);
        
            if (r == 0){
                addRandomHiddenToHiddenConnection();
            }
            else if (r == 1){
                addRandomHiddenToOutputConnection();
            }
            else if (r == 2){
                addRandomInputToHiddenConnection();
            }
            else{
                addRandomInputToOutputConnection();
            }
        
    }
    void removeRandomConnectionHiddens(){
        int randomHiddenIndex = randint(0, hiddens.size() - 1);
        if (hiddens[randomHiddenIndex].connections.size() > 0){
            int randomConnectionIndex = randint(0, hiddens[randomHiddenIndex].connections.size()-1);
            hiddens[randomHiddenIndex].connections.erase(hiddens[randomHiddenIndex].connections.begin()+randomConnectionIndex);
        }
    }
    void removeRandomConnectionOutputs(){
        int randomOutputsIndex = randint(0, outputs.size() - 1);
        if (outputs[randomOutputsIndex].connections.size() > 0){
            int randomConnectionIndex = randint(0, outputs[randomOutputsIndex].connections.size()-1);
            outputs[randomOutputsIndex].connections.erase(outputs[randomOutputsIndex].connections.begin()+randomConnectionIndex);
        }
    }
    void removeRandomConnection(){
        if (randf(0, 1) < 0.5){
            removeRandomConnectionHiddens();
        }
        else{
            removeRandomConnectionOutputs();
        }
    }
    void mutateRandomConnection(){
        if (randf(0, 1) < 0.5){
            int randomOutputsIndex = randint(0, outputs.size() - 1);
            if (outputs[randomOutputsIndex].connections.size() > 0){
                int randomConnectionIndex = randint(0, outputs[randomOutputsIndex].connections.size()-1);
                outputs[randomOutputsIndex].connections[randomConnectionIndex].weight += randf(-0.1, 0.1);
            }
        }
        else{
            int randomHiddenIndex = randint(0, hiddens.size() - 1);
            if (hiddens[randomHiddenIndex].connections.size() > 0){
                int randomConnectionIndex = randint(0, hiddens[randomHiddenIndex].connections.size()-1);
                hiddens[randomHiddenIndex].connections[randomConnectionIndex].weight += randf(-0.1, 0.1);
            }
        }
    }

    void weightedSumHiddens(){
        for (int i=0;i<hiddens.size();i++){        
            hiddens[i].value = 0;
            
            for (int j=0;j<hiddens[i].connections.size();j++){
                if (hiddens[i].connections[j].layerFrom == IN_){
                    hiddens[i].value += hiddens[i].connections[j].weight * inputs[hiddens[i].connections[j].indexFrom].value;
                }
                else if (hiddens[i].connections[j].layerFrom == HIDDEN_){
                    hiddens[i].value += hiddens[i].connections[j].weight * hiddens[hiddens[i].connections[j].indexFrom].value;
                }
            }
            hiddens[i].value += hiddens[i].bias;
            hiddens[i].value = activate(hiddens[i].value);   
        }
    }
    void weightedSumOutputs(){
        for (int i=0;i<outputs.size();i++){        
            outputs[i].value = 0;
            
            for (int j=0;j<outputs[i].connections.size();j++){
                if (outputs[i].connections[j].layerFrom == IN_){
                    outputs[i].value += outputs[i].connections[j].weight * inputs[outputs[i].connections[j].indexFrom].value;
                }
                else if (outputs[i].connections[j].layerFrom == HIDDEN_){
                    outputs[i].value += outputs[i].connections[j].weight * hiddens[outputs[i].connections[j].indexFrom].value;
                }
            }
            outputs[i].value += outputs[i].bias;
            outputs[i].value = activate(outputs[i].value);   
        }
    }
    void mutateBias(){
        if (randf(0, 1) < 0.5){
            int r = randint(0, outputs.size()-1);
            outputs[r].bias += randf(-0.1, 0.1);
        }
        else{
            int r = randint(0, hiddens.size()-1);
            hiddens[r].bias += randf(-0.1, 0.1);
        }
    }
    void mutate(float rate, int repeats, bool modifyHiddens=true){
        for (int z=0;z<repeats;z++){
            if (randf(0, 1) < rate / 2){
                addRandomConnection();
            }
            if (randf(0, 1)){
                mutateRandomConnection();
            }
            if (randf(0, 1) < rate / 4 && modifyHiddens){
                hiddens.push_back(Neuron());
            }
            if (randf(0, 1) < rate / 2){
                //removeRandomConnection();
            }
            if (randf(0, 1) < rate){
                mutateBias();
            }
        }
    }
    vector<float> feedForward(vector<float> ins){
        vector<float> outs;
        for (int i=0;i<inputs.size();i++){
            inputs[i].value = ins[i];
        }
        weightedSumHiddens();
        weightedSumOutputs();
        for (Neuron n : outputs){
            outs.push_back(n.value);
        }
        return outs;
    }
    void printNetwork(){
        int totalCons = 0;
        printf("\n");
        for (Neuron n : inputs){
            printf("- ");
            totalCons += n.connections.size();
        }
        printf("\n");
        for (Neuron n : hiddens){
            printf("- ");
            totalCons += n.connections.size();
        }
        printf("\n");
        for (Neuron n : outputs){
            printf("- ");
            totalCons += n.connections.size();
        }
        printf("\n%d Connections\n", totalCons);
    }
    Dynet copy(){
        Dynet r = Dynet(*this);
        return r;
    }
};

