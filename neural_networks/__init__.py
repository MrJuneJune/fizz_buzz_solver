import torch
import torch.nn as nn

from constants import NUM_DIGITS, NUMS_NEURONS, NUMBER_OF_KEYS

"""
FizzBuzz neural network objects.

It has two connected layers for learning and output.

Activation layer is using non-linear(ReLU) if it is just linear apparently, it cannot learn anything more complexed.
I actually don't know why the fuck it is that case lol
"""
class FizzBuzzNN(nn.Module):
    def __init__(self):
        super(FizzBuzzNN, self).__init__()
        # input sizes is the fixed bit sizes, output is 100 which can be adjusted.
        self.layer1 = nn.Linear(NUM_DIGITS, NUMS_NEURONS)
        self.layer2 = nn.Linear(NUMS_NEURONS, NUMBER_OF_KEYS)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

