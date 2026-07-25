import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from models.mrtf_model import MRTFModel

from training.trainer import MRTFTrainer

from training.loss_functions import MRTFLoss

from training.checkpoint import save_checkpoint




def train_mrtf(
    train_loader,
    val_loader,
    epochs=100
):


    device = (
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )



    model=MRTFModel().to(
        device
    )



    optimizer=AdamW(

        model.parameters(),

        lr=1e-4,

        weight_decay=1e-5

    )


    scheduler=CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )



    criterion=MRTFLoss()



    trainer=MRTFTrainer(

        model,

        optimizer,

        criterion,

        device

    )



    best_acc=0



    for epoch in range(
        epochs
    ):


        train_loss=trainer.train_epoch(
            train_loader
        )


        val_acc=trainer.validate(
            val_loader
        )


        scheduler.step()



        print(
            f"""
Epoch {epoch+1}/{epochs}

Training Loss:
{train_loss:.4f}

Validation Accuracy:
{val_acc:.4f}

"""
        )



        if val_acc > best_acc:


            best_acc=val_acc


            save_checkpoint(

                model,

                optimizer,

                epoch,

                train_loss,

                "checkpoints/best_mrtf.pt"

            )



    return model




if __name__=="__main__":


    print(
        "MRTF training pipeline initialized."
    )
