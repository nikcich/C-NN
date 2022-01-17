/*
Build/Run:
g++ main.cpp -o main
./main

Basic neural network implementation using vectors, no backprop or anything
**/

#include <iostream>
#include <math.h>
#include <vector>
#include <random>
#include <time.h>

using namespace std;

const int mutate_rate = 0.5;

double sigmoid(double x)
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

class Node
{
public:
    Node()
    {
        this->value = 0;
    }
    void setValue(double num)
    {
        this->value = num;
    }

    double getValue()
    {
        return this->value;
    }

private:
    double value;
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

class Layer
{
public:
    Layer()
    {
        this->size = 0;
    }

    Layer(int n)
    {
        this->size = n + 1;
        this->nextSize = 0;

        for (int i = 0; i < n + 1; i++)
        {
            Node *new_node = new Node();
            this->nodes.push_back(new_node);
        }

        this->nodes.at(this->size - 1)->setValue(1);
    }

    Layer(int n, int next)
    { // do connections on this oine
        this->size = n + 1;
        this->nextSize = next;

        for (int i = 0; i < n + 1; i++)
        {
            Node *new_node = new Node();
            this->nodes.push_back(new_node);
        }

        this->nodes.at(this->size - 1)->setValue(1);

        vector<vector<double>> weightsTemp;

        for (int i = 0; i < this->size; i++)
        {
            vector<double> blank(this->nextSize, 0.0);
            weightsTemp.push_back(blank);
        }

        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<double> distr(0, 1);

        for (int i = 0; i < this->size; i++)
        {
            for (int j = 0; j < next; j++)
            {
                weightsTemp[i][j] = distr(eng);
            }
        }

        this->weights = weightsTemp;
    }

    Layer(int size, int nSize, vector<vector<double>> weights){
        this->size = size;
        this->nextSize = nSize;
        for (int i = 0; i < size; i++)
        {
            Node *new_node = new Node();
            this->nodes.push_back(new_node);
        }
        this->nodes.at(this->size - 1)->setValue(1);

        this->weights = weights;
    }

    void input(vector<double> in)
    {
        for (int i = 0; i < in.size(); i++)
        {
            this->nodes[i]->setValue(in[i]);
        }
    }

    vector<double> forward()
    { // input for next layer or the results
        vector<double> results;

        for (int j = 0; j < this->nextSize; j++)
        {
            results.push_back(0.0);
        }

        for (int i = 0; i < this->size; i++)
        {
            for (int j = 0; j < this->nextSize; j++)
            {
                vector<vector<double>> w = this->weights;
                results[j] = results[j] + (w[i][j] * this->nodes[i]->getValue());
            }
        }

        for (int j = 0; j < this->nextSize; j++)
        {
            results[j] = sigmoid(results[j]);
        }

        return results;
    }

    void mutateWeights()
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<double> distr(0, 1);

        for (int i = 0; i < this->weights.size(); i++)
        {
            for (int j = 0; j < this->weights[i].size(); j++)
            {
                double num = distr(eng);

                if (num < mutate_rate)
                {
                    this->weights[i][j] = distr(eng);
                }
            }
        }
    }

    void mutateBias(){
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<double> distr(0, 1);

        double num = distr(eng);

        if (num < mutate_rate)
        {
            Node* bias = this->nodes[this->nodes.size()-1]; // Last node
            bias->setValue(distr(eng));
        }
    }

    vector<vector<double>> copyWeights(){
        return this->weights;
    }

    int getSize(){
        return this->size;
    }

    int getNextSize(){
        return this->nextSize;
    }

    double copyBiasValue(){
        return this->nodes[this->nodes.size()-1]->getValue();
    }

    void setWeight(vector<vector<double>> w){
        this->weights = w;
    }

    void setBias(double n){
        this->nodes[this->nodes.size()-1]->setValue(n);
    }

private:
    int size;
    int nextSize;
    vector<Node *> nodes;
    vector<vector<double>> weights;
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

class Network
{

public:
    Network(vector<int> sizes)
    {

        for (int i = 0; i < sizes.size(); i++)
        { // for each layer we want
            int layerSize = sizes[i];
            Layer *new_layer;
            if (i < sizes.size() - 1)
            {
                new_layer = new Layer(layerSize, sizes[i + 1]); // For connections
            }
            else
            {
                new_layer = new Layer(layerSize);
            }

            this->layers.push_back(new_layer);
        }
    }

    void input(vector<double> in)
    {
        this->layers.at(0)->input(in);
    }

    vector<double> activate()
    {
        vector<double> results;

        for (int i = 0; i < this->layers.size(); i++)
        {

            Layer *curr = this->layers[i];

            if (i < layers.size() - 1)
            {
                results = curr->forward();
                this->layers[i + 1]->input(results);
            }
        }

        return results;
    }

    Network(vector<Layer*> lyrs){
        this->layers = lyrs;
    }

    Network(Network* old){
        vector<Layer*> lyr;

        for(int i = 0; i < this->layers.size()-1; i++){
            vector<vector<double>> w = old->layers[i]->copyWeights();
            int size = old->layers[i]->getSize();
            int nSize = old->layers[i]->getNextSize();

            Layer* newLyr = new Layer(size, nSize, w);

            lyr.push_back(newLyr);
        }

        Layer* lastLyr = new Layer(this->layers[this->layers.size()-1]->getSize());
        lyr.push_back(lastLyr);

        this->layers = lyr;
    }

    void mutate()
    {

        for (int i = 0; i < this->layers.size() - 1; i++)
        { // - 1 because not the last layer since has no weights

            Layer *curr = this->layers[i];

            curr->mutateWeights();
            curr->mutateBias();
        }
    }

    double error = 0.0;

    Network* clone(){
        return new Network(*this);
    }

    void combine(Network* other){
        vector<Layer*> lyrClone;

        for(int i = 0; i < this->layers.size(); i++){
            cout << i << endl;
            Layer* curr = this->layers[i];
            int siz = curr->getSize();
            int nSiz = curr->getNextSize();
            
            vector<vector<double>> wgt;
            double biasV;

            if(i%2==0){ // Use this->layers
                if (i < this->layers.size() - 1){
                    wgt = curr->copyWeights();
                    biasV = curr->copyBiasValue();
                }
            }else{ // use other->layers
                if (i < this->layers.size() - 1){
                    wgt = other->layers[i]->copyWeights();
                    biasV = other->layers[i]->copyBiasValue();
                }
            }


            Layer* newLayer;

            if (i < this->layers.size() - 1)
            {
                newLayer = new Layer(siz, nSiz);
                newLayer->setWeight(wgt);
                newLayer->setBias(biasV);
            }
            else
            {
                newLayer = new Layer(siz);
            }

            lyrClone.push_back(newLayer);
        }

        this->layers = lyrClone;
    }

    bool combined = false;
private:
    vector<Layer *> layers;
    
};

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

int main()
{

    vector<int> layerData = {2, 4, 4, 1};
    vector<double> inputData = {2.0, 2.0};

    vector<Network *> pop;
    int popSize = 1000;
    int generations = 100;

    for (int i = 0; i < popSize; i++)
    {
        pop.push_back(new Network(layerData));
    }

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0, 10);
    //std::uniform_real_distribution<int> intDistr(0, 10000);

    for(int i = 0; i < generations; i++){

        // Run the feed forward and get results with current values
        int bestIdx = 0;
        double bestErr = 100000;

        double cumErr = 0.0;

        for(int j = 0; j < popSize; j++){
            double n1 = 2.0;
            double n2 = 2.0;
            vector<double> ins = {n1, n2};

            Network *curr = pop[j];

            curr->input(ins);
            vector<double> results = curr->activate();

            curr->error = abs(results[0] - (n1 * n2));

            if(curr->error < bestErr){
                bestIdx = j;
                bestErr = curr->error;
            }
            cumErr += curr->error;
        }

        for(int j = 0; j < popSize; j++){
            if(j != bestIdx){
                pop[j] = pop[bestIdx]->clone();

                pop[j]->mutate();
                pop[j]->mutate();
            }
        }

        cout << "Generation #" << i << endl;
        cout << "Average error: " << cumErr/(popSize*1.0) << endl;
        cout << "Best Error: " << bestErr << endl;

    }

    
    

    return 0;
}
