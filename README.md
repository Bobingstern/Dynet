# Dynet
A concept I came up which ditches the idea of "layers" in a neural network.

![A picture of the XOR test's error graph](assets/XOR_.png)

## Install
Copy [`Dynet.py`](Dynet.py) to your project.

## Run the example

Install `matplotlib` with `pip install matplotlib` to run the example in 
[`main.py`](main.py).

## How it works
Classic neural networks use layers as a way of organizing neurons. 
"Dynet" uses a single layers to process inputs and outputs where neurons can 
directly connect to outputs or pass through mutliple neurons and even connect to themselves
