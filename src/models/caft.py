import torch
import torch.nn as nn



class CrossAttentionFusionTransformer(nn.Module):


    def __init__(
        self,
        embed_dim=512,
        heads=8,
        layers=4
    ):

        super().__init__()


        encoder_layer=nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=heads,

            batch_first=True,

            dropout=0.1

        )


        self.caft=nn.TransformerEncoder(

            encoder_layer,

            num_layers=layers

        )



    def forward(
        self,
        modality_tokens
    ):


        """
        Input:

        modality_tokens:

        [B,3,512]

        Voice
        MRI
        Sensor

        """


        fused=self.caft(
            modality_tokens
        )


        return fused.mean(dim=1)
