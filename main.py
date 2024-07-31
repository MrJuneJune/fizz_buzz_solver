import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Don't want to write if elses
numbers = {3: 1, 5: 2}

# Universe start from 0
NUMBER_OF_KEYS = sum(numbers.values()) + 1 

# sizes of bits (vectors)
NUM_DIGITS = 10

# Number of neurons
NUMS_NEURONS = 100

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

"""
Function to print out all weigths and biases
"""
def print_layer_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}:\n{param.data.numpy()}\n')

"""
Solve fizz buzz and just put it as 1 x SUMS matrixs
"""
def fizz_buzz_solver(num):
    curr_index = 0
    ans = [0 for _ in range(NUMBER_OF_KEYS)]
    for  number, index in numbers.items():
        if num % number == 0:
            curr_index += index
    ans[curr_index] = 1
    return  ans

"""
Print Fizz buzz from the perdictions
"""
def fizz_buzz(i, prediction):
    return [str(i), "Fizz", "Buzz", "FizzBuzz"][prediction]

def find(lis):
    for key, i in enumerate(lis):
        if i == 1:
            return key

"""
Function to calculate accuracy.

argmax of outputs will grab indexes of max values from the inputs.
Since our inputs (outputs from the models) will be 1 x 4 i.e) [0, 0, 1, 0] the indices.
"""
def calculate_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == torch.argmax(y, dim=1)).sum().item()
        accuracy = correct / len(y)
    model.train()
    return accuracy

# double checking my code
# for x in range(1, 100):
#     print(fizz_buzz(x, find(fizz_buzz_solver(x))))

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def create_datasets():
    x_train = torch.Tensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 1024)]))
    y_train = torch.Tensor(np.array([fizz_buzz_solver(i) for i in range(101, 1024)]))

    return x_train, y_train

def train(model, x_train, y_train, loss_fn, optimizer, epochs=5000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = loss_fn(outputs, torch.argmax(y_train, dim=1)) # check the differences
        loss.backward() # backwards to compute gradients
        optimizer.step() # update the weights 

        if epoch % 500 == 0:
            accuracy = calculate_accuracy(model, x_train, y_train)
            print_layer_parameters(model)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%')



model = FizzBuzzNN()
x_train, y_train = create_datasets()

loss_fn = nn.CrossEntropyLoss()

# this is used to change the weights of the nn
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, x_train, y_train, loss_fn, optimizer)

x_test = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
x_test = torch.Tensor(x_test)

model.eval()
with torch.no_grad():
    test_outputs = model(x_test)
    predictions = torch.argmax(test_outputs, dim=1).numpy()

# for i in range(1, 101):
#     print(fizz_buzz(i, predictions[i-1]))
