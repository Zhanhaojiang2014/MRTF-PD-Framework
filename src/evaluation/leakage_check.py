import numpy as np



def check_subject_overlap(
    train_ids,
    val_ids,
    test_ids
):


    train=set(train_ids)

    val=set(val_ids)

    test=set(test_ids)



    overlap = {

        "train_val":
        len(train & val),

        "train_test":
        len(train & test),

        "val_test":
        len(val & test)

    }


    return overlap





def verify_no_leakage(
    train_ids,
    val_ids,
    test_ids
):


    overlap=check_subject_overlap(

        train_ids,

        val_ids,

        test_ids

    )


    return all(

        value==0

        for value in overlap.values()

    )





def leakage_report():

    return {

        "subject_split":
        "Subject-wise",

        "normalization":
        "Training only",

        "GAN augmentation":
        "Training only",

        "latent_alignment":
        "Training embeddings only",

        "threshold":
        "Validation only"

    }
