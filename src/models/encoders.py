import torch
import torch.nn as nn


# ============================================================
# Voice Encoder
# CNN + BiLSTM
# ============================================================

class VoiceEncoder(nn.Module):

    def __init__(
        self,
        input_dim=128,
        hidden_dim=256
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                input_dim,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()

        )


        self.bilstm = nn.LSTM(

            input_size=128,

            hidden_size=hidden_dim,

            num_layers=2,

            batch_first=True,

            bidirectional=True

        )


        self.projection = nn.Linear(
            hidden_dim*2,
            512
        )


    def forward(self,x):

        x = self.cnn(x)

        x = x.transpose(1,2)

        output,_ = self.bilstm(x)

        embedding = output[:,-1,:]

        return self.projection(embedding)



# ============================================================
# MRI Vision Transformer Encoder
# ============================================================


class MRIEncoder(nn.Module):

    def __init__(
        self,
        img_size=224,
        embed_dim=512,
        heads=8,
        layers=6
    ):

        super().__init__()


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=heads,

            batch_first=True

        )


        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=layers

        )


        self.patch_projection = nn.Linear(
            16*16,
            embed_dim
        )


    def forward(self,x):

        B,C,H,W=x.shape


        patches = x.unfold(
            2,16,16
        ).unfold(
            3,16,16
        )


        patches = patches.reshape(
            B,
            -1,
            256
        )


        tokens=self.patch_projection(
            patches
        )


        encoded=self.transformer(tokens)


        return encoded.mean(dim=1)




# ============================================================
# Sensor Temporal Transformer
# ============================================================


class SensorEncoder(nn.Module):

    def __init__(
        self,
        input_dim=6,
        embed_dim=512,
        heads=8
    ):

        super().__init__()


        self.embedding=nn.Linear(
            input_dim,
            embed_dim
        )


        layer=nn.TransformerEncoderLayer(

            d_model=embed_dim,

            nhead=heads,

            batch_first=True

        )


        self.transformer=nn.TransformerEncoder(

            layer,

            num_layers=4

        )



    def forward(self,x):

        x=self.embedding(x)

        x=self.transformer(x)

        return x.mean(dim=1)
