"""
PPMI MRI dataset preparation script.

PPMI requires approved access.
"""


import os



def check_ppmi(path):


    if os.path.exists(path):

        print(
            "PPMI dataset detected."
        )


    else:

        print(
            """
PPMI dataset is missing.

Please request access from
Parkinson's Progression Markers Initiative
and download required MRI data.
"""
        )



if __name__=="__main__":


    check_ppmi(
        "datasets/raw/ppmi"
    )
