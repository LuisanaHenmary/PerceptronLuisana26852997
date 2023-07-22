from perceptron import Perceptron
from utils import get_dataframe


def separate_float_and_non_float_processed(all_data):
    """Prepare the array of entries and classify whether or not float processed.
       1 If it's float processed.
       0 If it's not float processed.

       :param all_data: It is the dataframe.
       :return: Inputs and desired outputs for the perceptron
       """

    only_window_glass = all_data[all_data["Type"] < 5]

    inputs = only_window_glass.drop(columns=[
        "Type",
        "Id"
    ]).values

    outputs = only_window_glass["Type"].values % 2 == 1

    return inputs, outputs


if __name__ == '__main__':

    my_data = get_dataframe()

    X, y = separate_float_and_non_float_processed(my_data)

    perceptron_neuron = Perceptron(X, y, columns_number=X.shape[1], epoch_num=200)

    print(perceptron_neuron.weight)
    print()

    perceptron_neuron.fit()

    print(perceptron_neuron.weight)