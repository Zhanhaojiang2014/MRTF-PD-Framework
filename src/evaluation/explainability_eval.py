import numpy as np



def explainability_confidence_score(
    shap_values
):

    """
    ECS calculation.

    Higher value indicates
    stable explanations.
    """


    variance=np.var(
        shap_values,
        axis=0
    )


    ecs=1-np.mean(
        variance
    )


    return float(
        np.clip(
            ecs,
            0,
            1
        )
    )





def explanation_stability(
    explanations_a,
    explanations_b
):


    similarity=np.corrcoef(

        explanations_a,

        explanations_b

    )[0,1]


    return similarity





def spearman_explanation_correlation(
    performance,
    ecs
):


    from scipy.stats import spearmanr


    rho,p=spearmanr(
        performance,
        ecs
    )


    return {

        "rho":
        rho,

        "p_value":
        p

    }




def counterfactual_distance(
    original,
    counterfactual
):


    return np.linalg.norm(

        original-counterfactual

    )
