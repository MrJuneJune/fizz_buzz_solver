"""
File with one of functions.
I didn't type any of these, but models are nn.models where x (input), and y(output) are the training data
"""
import torch

"""
Function to print out all weigths and biases
"""
def print_layer_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}:\n{param.data.numpy()}\n')


"""
Simple function to decode output
"""
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
