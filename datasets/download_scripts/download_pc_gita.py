"""
PC-GITA dataset preparation script.

The dataset must be manually downloaded
according to the official access policy.
"""


import os



DATASET_NAME = "PC-GITA"


def check_dataset(path):

    if os.path.exists(path):

        print(
            "PC-GITA dataset found."
        )

    else:

        print(
            """
PC-GITA dataset not found.

Please download it from the official repository
and place it in:

datasets/raw/pc_gita/
"""
        )



if __name__ == "__main__":


    check_dataset(
        "datasets/raw/pc_gita"
    )
