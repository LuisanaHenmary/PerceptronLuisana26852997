import numpy as np


class Perceptron:
    """This class is for creating a perceptron neuron.

        Attributes:
        weight (numpy.ndarray): Synaptic weights
        columns_number (int): The number of columns of each entry.
        inputs (numpy.ndarray): Neuron inputs.
        outputs (numpy.ndarray): The desired outputs.
        epoch_num (int): The number of epochs.
        learning_rate (float): The learning rate.
    """

    def __init__(self, inputs, outputs, columns_number=2, epoch_num=20, lr=0.01):

        """Initialize the neuron.

        :param inputs: Neuron inputs
        :param outputs: The desired outputs.
        :param columns_number: The number of columns of each entry, default value 2.
        :param epoch_num: The number of epochs, default value 20.
        :param lr: The learning rate, default value 2.
        """

        # self.weight = np.random.random(columns_number + 1)/np.sqrt(columns_number)
        self.weight = np.zeros(columns_number + 1)

        self.columns_number = columns_number
        self.inputs = inputs
        self.outputs = outputs
        self.epoch_num = epoch_num
        self.learning_rate = lr

    @staticmethod
    def step(n):
        """It is the activation function, step.

        :param n: It is the net input to the neuron.
        :return: 1 if the net input is greater than or equal to zero, otherwise returns 0.
        """
        return 1 if n >= 0 else 0

    def fit(self):
        """It is to obtain the synaptic weights.

        :return: The resulting synaptic weights.
        """

        self.inputs = np.c_[self.inputs, np.ones((self.inputs.shape[0]))]

        for _ in np.arange(self.epoch_num):
            for (xi, desire_output) in zip(self.inputs, self.outputs):
                output = self.step(np.dot(xi, self.weight))

                if output != desire_output:
                    error = output - desire_output

                    self.weight += -self.learning_rate * error * xi
