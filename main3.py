from perceptron import Perceptron
from utils import get_dataframe


def separate_building_and_vehicle_window(all_data):
    """Prepare the array of entries and classify whether it is a vehicle window or building window.
       1 If it's building window.
       0 If it's vehicle window.

       :param all_data: It is the dataframe.
       :return: Inputs and desired outputs for the perceptron
       """
    only_window_glass = all_data[all_data["Type"] < 5]

    inputs = only_window_glass.drop(columns=[
        "Type",
        "Id"
    ]).values

    outputs = only_window_glass["Type"].values < 3

    return inputs, outputs


if __name__ == '__main__':

    my_data = get_dataframe()

    X, y = separate_building_and_vehicle_window(my_data)

    perceptron_neuron = Perceptron(X, y, columns_number=X.shape[1], epoch_num=200)

    print(perceptron_neuron.weight)
    print()

    perceptron_neuron.fit()

    print(perceptron_neuron.weight)