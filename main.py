from perceptron import Perceptron
from utils import get_dataframe


def separate_window_and_non_window_glass(all_data):
    """Prepare the array of entries and classify whether or not it is a window glass.
    1 If it's window glass.
    0 If it's not window glass.

    :param all_data: It is the dataframe.
    :return: Inputs and desired outputs for the perceptron
    """

    inputs = all_data.drop(columns=[
        "Type",
        "Id"
    ]).values

    outputs = all_data["Type"].values < 5

    return inputs, outputs


if __name__ == '__main__':

    my_data = get_dataframe()

    X, y = separate_window_and_non_window_glass(my_data)

    perceptron_neuron = Perceptron(X, y, columns_number=X.shape[1], epoch_num=200)

    print(perceptron_neuron.weight)
    print()

    perceptron_neuron.fit()

    print(perceptron_neuron.weight)
