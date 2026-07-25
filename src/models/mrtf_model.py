import torch
import torch.nn as nn

from .encoders import (
    VoiceEncoder,
    MRIEncoder,
    SensorEncoder
)

from .caft import (
    CrossAttentionFusionTransformer
)



class MRTF(nn.Module):


    def __init__(self):

        super().__init__()


        self.voice_encoder=VoiceEncoder()

        self.mri_encoder=MRIEncoder()

        self.sensor_encoder=SensorEncoder()



        self.caft=CrossAttentionFusionTransformer()



        self.classifier=nn.Sequential(

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(256,64),

            nn.ReLU(),


            nn.Linear(64,2)

        )



    def forward(
        self,
        voice,
        mri,
        sensor
    ):


        z_v=self.voice_encoder(voice)

        z_m=self.mri_encoder(mri)

        z_s=self.sensor_encoder(sensor)



        tokens=torch.stack(

            [
                z_v,
                z_m,
                z_s
            ],

            dim=1

        )


        fused=self.caft(tokens)


        output=self.classifier(
            fused
        )


        return {

            "prediction":output,

            "latent":fused,

            "tokens":tokens

        }
