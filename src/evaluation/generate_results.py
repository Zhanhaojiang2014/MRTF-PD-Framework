import pandas as pd


from metrics import classification_metrics

from statistical_tests import (
    bootstrap_ci,
    cohens_d,
    cliffs_delta
)




def generate_table_results(
    predictions,
    output_path
):


    rows=[]


    for model,data in predictions.items():


        metrics=classification_metrics(

            data["y_true"],

            data["y_pred"],

            data["y_prob"]

        )


        metrics["Model"]=model


        rows.append(
            metrics
        )



    df=pd.DataFrame(
        rows
    )


    df.to_csv(
        output_path,
        index=False
    )


    return df





def generate_statistical_summary(
    mrtf_scores,
    baseline_scores
):


    return {


        "95_CI":
        bootstrap_ci(
            mrtf_scores
        ),


        "Cohens_d":
        cohens_d(
            mrtf_scores,
            baseline_scores
        ),


        "Cliffs_delta":
        cliffs_delta(
            mrtf_scores,
            baseline_scores
        )

    }




if __name__=="__main__":


    print(
        "MRTF evaluation pipeline initialized."
    )
