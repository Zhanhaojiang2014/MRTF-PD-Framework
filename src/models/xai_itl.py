import shap
import torch



class XAI_ITL:


    def __init__(
        self,
        model,
        background
    ):

        self.model=model

        self.explainer=shap.DeepExplainer(

            model,

            background

        )



    def shap_explanation(
        self,
        x
    ):


        values=self.explainer.shap_values(x)

        return values



    def counterfactual_loss(
        self,
        original,
        counterfactual
    ):


        return torch.mean(

            torch.abs(
                original-counterfactual
            )

        )



    def ECS(
        self,
        attribution
    ):

        """

        Explanation Confidence Score

        Internal consistency metric

        """

        score=torch.mean(

            torch.abs(attribution)

        )


        return score.item()
