import torch
import torch.nn as nn



class PolicyNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(3,128),

            nn.ReLU(),

            nn.Linear(128,64),

            nn.ReLU(),

            nn.Linear(64,3)

        )


    def forward(self,state):

        return self.network(state)



class ReinforcementAssistedSelfLearning:


    def __init__(
        self,
        lr=1e-4
    ):

        self.policy=PolicyNetwork()

        self.optimizer=torch.optim.Adam(

            self.policy.parameters(),

            lr=lr

        )


    def compute_reward(
        self,
        accuracy,
        ecs,
        loss
    ):

        return (

            0.5*accuracy +

            0.3*ecs -

            0.2*loss

        )



    def update(
        self,
        state,
        reward
    ):


        action=self.policy(state)


        loss=-torch.mean(
            action*reward
        )


        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        return loss.item()
