# FizzBuzz Solvers

## Motivation

FizzBuzz is a common problem in programming interviews. I decided to create a neural network (NN) that solves FizzBuzz using Python for fun. This way, if someone asks me this question, I can demonstrate a solution using a neural network and say, "I don't know how it works" humorously.

I also plan to add a graphical implementation to show how these NNs work, which will be an interesting challenge. My goal is to visualize this on the web, either by replicating the whole thing in JavaScript or by using a Python API to show updates.

## Prerequisites

Set up any Python environment. I usually create a Docker image or a Python virtual environment.

```bash
pip install -r requirements.txt
```

## How to Run

Run the `main.py` script:

```bash
mkdir models  # create directory to store models in
python main.py # or any equivalent command
```

## Explanations

The initial input vector is a bitized integer, defined by `NUM_DIGITS = 20` (in hindsight, I should have called it bit length). The model is trained on numbers ranging from 2000 to 100000. After 10000 iterations in a single layer with two NNs, each having 100 neurons, the model achieved a score of 93% on numbers it had never seen before, which is not great. I could probably improve the performance by increasing the matrix sizes or the number of training iterations, as the loss was still decreasing. However, for now, this is fine for a fun, simple project. I plan to add a frontend and make it interactive online to visualize the weights and show how the model gets trained.
