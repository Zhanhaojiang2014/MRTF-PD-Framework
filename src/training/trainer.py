import torch
from tqdm import tqdm



class MRTFTrainer:


    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        device="cuda"
    ):


        self.model=model

        self.optimizer=optimizer

        self.loss_function=loss_function

        self.device=device



    def train_epoch(
        self,
        dataloader
    ):


        self.model.train()


        total_loss=0



        for batch in tqdm(
            dataloader
        ):


            self.optimizer.zero_grad()



            outputs = self.model(
                batch
            )



            losses=self.loss_function(
                outputs["prediction"],
                batch["label"],
                outputs["latent"],
                outputs["ecs"],
                outputs["reward"]
            )


            loss=losses["total_loss"]


            loss.backward()


            self.optimizer.step()



            total_loss += loss.item()



        return (
            total_loss /
            len(dataloader)
        )



    @torch.no_grad()
    def validate(
        self,
        dataloader
    ):


        self.model.eval()


        correct=0

        total=0



        for batch in dataloader:


            outputs=self.model(
                batch
            )


            prediction=torch.argmax(
                outputs["prediction"],
                dim=1
            )


            correct += (
                prediction ==
                batch["label"].to(self.device)
            ).sum().item()



            total += (
                batch["label"].size(0)
            )



        accuracy = (
            correct /
            total
        )


        return accuracy
