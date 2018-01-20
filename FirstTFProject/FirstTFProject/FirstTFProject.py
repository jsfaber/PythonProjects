## Jasper Faber - 2018
## First tries with tensorflow
## https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # Import data
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)



# Start main
if __name__ == "__main__":
    main()
