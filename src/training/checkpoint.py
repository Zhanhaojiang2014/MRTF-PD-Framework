import torch
import os



def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):

    """
    Save MRTF training state.
    """


    checkpoint = {

        "epoch":
        epoch,

        "model_state":
        model.state_dict(),

        "optimizer_state":
        optimizer.state_dict(),

        "loss":
        loss

    }


    torch.save(
        checkpoint,
        path
    )





def load_checkpoint(
    model,
    optimizer,
    path,
    device="cuda"
):


    checkpoint=torch.load(
        path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state"]
    )


    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )


    return (

        model,

        optimizer,

        checkpoint["epoch"],

        checkpoint["loss"]

    )
