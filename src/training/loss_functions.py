import torch
import torch.nn as nn
import torch.nn.functional as F



class MRTFLoss(nn.Module):

    """
    Multi-objective loss function for MRTF.

    Components:

    L_total =
        L_cls
        + λ1 L_align
        + λ2 L_xai
        + λ3 L_rasl

    """

    def __init__(
        self,
        lambda_align=0.3,
        lambda_xai=0.2,
        lambda_rasl=0.1
    ):

        super().__init__()

        self.lambda_align = lambda_align
        self.lambda_xai = lambda_xai
        self.lambda_rasl = lambda_rasl


        self.classification_loss = (
            nn.CrossEntropyLoss()
        )



    def classification_loss_fn(
        self,
        prediction,
        target
    ):

        return self.classification_loss(
            prediction,
            target
        )



    def latent_alignment_loss(
        self,
        z_voice,
        z_mri,
        z_sensor
    ):

        """
        Alignment between modality embeddings.
        """

        loss_vm = F.mse_loss(
            z_voice,
            z_mri
        )

        loss_vs = F.mse_loss(
            z_voice,
            z_sensor
        )

        loss_ms = F.mse_loss(
            z_mri,
            z_sensor
        )


        return (
            loss_vm +
            loss_vs +
            loss_ms
        ) / 3



    def explainability_loss(
        self,
        explanation_score
    ):

        """
        Encourage stable explanations.

        ECS close to 1.
        """

        return torch.mean(
            (1-explanation_score)**2
        )



    def rasl_loss(
        self,
        reward
    ):

        """
        Reinforcement-assisted optimization loss.
        """

        return -torch.mean(
            reward
        )



    def forward(
        self,
        outputs,
        labels,
        latent_features,
        ecs,
        reward
    ):


        classification = self.classification_loss_fn(
            outputs,
            labels
        )


        alignment = self.latent_alignment_loss(
            latent_features["voice"],
            latent_features["mri"],
            latent_features["sensor"]
        )


        xai = self.explainability_loss(
            ecs
        )


        rasl = self.rasl_loss(
            reward
        )


        total = (

            classification

            +
            self.lambda_align * alignment

            +
            self.lambda_xai * xai

            +
            self.lambda_rasl * rasl

        )


        return {

            "total_loss": total,

            "classification_loss":
            classification,

            "alignment_loss":
            alignment,

            "xai_loss":
            xai,

            "rasl_loss":
            rasl
        }
