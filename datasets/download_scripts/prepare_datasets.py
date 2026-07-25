import os



def create_dataset_structure():


    folders=[

        "datasets/raw",

        "datasets/raw/pc_gita",

        "datasets/raw/ppmi",

        "datasets/raw/daphnet",

        "datasets/processed/voice",

        "datasets/processed/mri",

        "datasets/processed/sensor"

    ]


    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


    print(
        "Dataset directory structure created."
    )



if __name__=="__main__":

    create_dataset_structure()
