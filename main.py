import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from helpers import calculate_accuracy, find
from constants import LEARN_END_INT, LEARN_RANGE, NUMBERS, NUMBER_OF_KEYS, NUM_DIGITS, MODEL_SAVE_PATH, LEARN_START_INT
from neural_networks import FizzBuzzNN

"""
Solve fizz buzz and put it as 1 x SUMS matrixs.

Lengths are dependent on the size of the dictionary. Made it generic so we can use more than 3, 5 and more close to N
"""
def fizz_buzz_solver(num):
    curr_index = 0
    ans = [0 for _ in range(NUMBER_OF_KEYS)]
    for  number, index in NUMBERS.items():
        if num % number == 0:
            curr_index += index[0]
    ans[curr_index] = 1
    return  ans

"""
Print Fizz buzz from the perdictions
"""
def fizz_buzz(i, prediction):
    # TODO(1): Update this shit as well
    return [str(i), "Fizz", "Buzz", "FizzBuzz"][prediction]

# double checking my code because I am dumb
# for x in range(1, 100):
#     print(fizz_buzz(x, find(fizz_buzz_solver(x))))

"""
Need to set input vector to be constants.
"""
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

"""
Creating a x(input), y(result) data sets from 101, to 1024  we can pick any number
"""
def create_datasets():
    print(f"Created a dataset from {LEARN_START_INT} to {LEARN_END_INT}")
    x_train = torch.Tensor(np.array([binary_encode(i, NUM_DIGITS) for i in LEARN_RANGE]))
    y_train = torch.Tensor(np.array([fizz_buzz_solver(i) for i in LEARN_RANGE]))

    return x_train, y_train

"""
Training
"""
def train(model, x_train, y_train, loss_fn, optimizer, epochs=10000):
    print("Start training!")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = loss_fn(outputs, torch.argmax(y_train, dim=1)) # check the differences
        loss.backward() # backwards to compute gradients
        optimizer.step() # update the weights 

        if epoch % 500 == 0:
            accuracy = calculate_accuracy(model, x_train, y_train)
            # Checking how it is getting updated.
            # print_layer_parameters(model)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%\nSaving the model...')
            torch.save(model.state_dict(), MODEL_SAVE_PATH + '/fizz_buzz_model.pth')
            print(f"Saved")


if __name__=="__main__":
    model = FizzBuzzNN()

    # Model save path
    import os
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    fizz_buzz_model = os.path.join(MODEL_SAVE_PATH, 'fizz_buzz_model.pth')

    # Load weights if available
    if os.path.exists(fizz_buzz_model):
        model.load_state_dict(torch.load(fizz_buzz_model, weights_only=True))
        print("Loaded generator weights")
    else:
        print("Creating a new model")
    
    x_train, y_train = create_datasets()
   
    loss_fn = nn.CrossEntropyLoss()
    
    # this is used to change the weights of the nn
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(model, x_train, y_train, loss_fn, optimizer)
   
    # Creating data to test model on that isn't from our data sets
    start_int = random.randint(0, LEARN_START_INT - 500)
    end_int = random.randint(start_int+100, start_int+300)

    print(f"Testing random int from {start_int} to {end_int}")
    x_test = np.array([binary_encode(i, NUM_DIGITS) for i in range(start_int, end_int)])
    x_test = torch.Tensor(x_test)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        predictions = torch.argmax(test_outputs, dim=1).numpy()
    
    wrong = 0
    start = 0
    for i in range(start_int, end_int):
        fake = fizz_buzz(i, predictions[start])
        real = fizz_buzz(i, find(fizz_buzz_solver(i)))
        wrong += 1 if fake != real else 0
        start += 1
        print(f"Model: {fake},\t Real: {real}")

    print(f"It was wrong {wrong} times. It has accuracy of {100 - round((wrong / start)*100)}%")
