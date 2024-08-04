# Don't want to write if elses
# TODO(1): probably can imporve this so we can add these keywards whichout updating the methods
NUMBERS = {3: [1, "Fizz"], 5: [2, "Buzz"]}

# Universe start from 0
# NUMBER_OF_KEYS = sum(map(lambda x : x[0], NUMBERS.values())) + 1 
NUMBER_OF_KEYS = 4 

# sizes of bits (vectors)
# 2^x since only positive.
NUM_DIGITS = 20 

# Number of neurons
NUMS_NEURONS = 100

# Saving model
MODEL_SAVE_PATH = "./models"

# Learning range
LEARN_START_INT = 2000
LEARN_END_INT = 100000
LEARN_RANGE = range(LEARN_START_INT, LEARN_END_INT)
