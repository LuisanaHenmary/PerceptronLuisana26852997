import pandas as pd
import numpy as np


def get_dataframe():

    """From a file with the extension data, it creates a dataframe.

    :return: A dataframe.
    """

    cases = []
    with open("dataset/glass.data", "r") as f:

        for line in f:
            cases.append(line.split(","))
        f.close()

    data = {
        "Id": np.array([x[0] for x in cases], dtype=int),
        "RI": np.array([x[1] for x in cases], dtype=float),
        "Na": np.array([x[2] for x in cases], dtype=float),
        "Mg": np.array([x[3] for x in cases], dtype=float),
        "Al": np.array([x[4] for x in cases], dtype=float),
        "Si": np.array([x[5] for x in cases], dtype=float),
        "K": np.array([x[6] for x in cases], dtype=float),
        "Ca": np.array([x[7] for x in cases], dtype=float),
        "Ba": np.array([x[8] for x in cases], dtype=float),
        "Fe": np.array([x[9] for x in cases], dtype=float),
        "Type": np.array([x[10] for x in cases], dtype=int)

    }

    data_frame = pd.DataFrame(data)

    print(data_frame)

    return data_frame
