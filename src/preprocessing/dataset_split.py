import numpy as np


class SubjectWiseSplitter:


    """
    Leakage-controlled dataset splitting.

    Train: 70%
    Validation: 15%
    Test: 15%

    Subject-level separation.
    """



    def __init__(
        self,
        seed=42
    ):

        self.seed=seed



    def split(
        self,
        subjects
    ):

        np.random.seed(
            self.seed
        )


        subjects=np.array(
            subjects
        )


        np.random.shuffle(
            subjects
        )


        n=len(subjects)


        train_end=int(
            0.7*n
        )


        val_end=int(
            0.85*n
        )


        train=subjects[:train_end]

        validation=subjects[
            train_end:val_end
        ]

        test=subjects[
            val_end:
        ]


        return {

            "train":
            train.tolist(),

            "validation":
            validation.tolist(),

            "test":
            test.tolist()

        }
