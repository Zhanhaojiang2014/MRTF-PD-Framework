"""
Daphnet Freezing-of-Gait dataset checker.
"""


import os



def check_daphnet(path):


    if os.path.exists(path):

        print(
            "Daphnet dataset available."
        )


    else:

        print(
            """
Daphnet dataset not found.

Download the dataset from UCI repository
and place files in:

datasets/raw/daphnet/
"""
        )



if __name__=="__main__":

    check_daphnet(
        "datasets/raw/daphnet"
    )
