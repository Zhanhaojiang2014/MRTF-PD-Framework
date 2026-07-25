import torch
import torch.nn as nn


class LatentFeatureAlignment(nn.Module):


    """
    Cross-cohort latent-space alignment.

    No subject-level matching is performed.

    Projection:
    modality embedding -> 512 dimension
    """



    def __init__(
        self,
        input_dim,
        latent_dim=512
    ):

        super().__init__()


        self.layer_norm=nn.LayerNorm(
            input_dim
        )


        self.projection=nn.Linear(
            input_dim,
            latent_dim
        )



    def normalize(
        self,
        x
    ):

        return self.layer_norm(x)



    def forward(
        self,
        x
    ):

        x=self.normalize(x)


        z=self.projection(x)


        return z
