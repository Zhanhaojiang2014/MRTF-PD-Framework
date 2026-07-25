import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)



def classification_metrics(
    y_true,
    y_pred,
    y_prob
):

    """
    Calculate MRTF evaluation metrics.
    """


    results = {

        "Accuracy":
        accuracy_score(
            y_true,
            y_pred
        ),


        "Precision":
        precision_score(
            y_true,
            y_pred
        ),


        "Recall":
        recall_score(
            y_true,
            y_pred
        ),


        "F1":
        f1_score(
            y_true,
            y_pred
        ),


        "AUC":
        roc_auc_score(
            y_true,
            y_prob
        )

    }


    return results



def confusion_metrics(
    y_true,
    y_pred
):


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()


    sensitivity = tp/(tp+fn)

    specificity = tn/(tn+fp)



    return {

        "Sensitivity":
        sensitivity,

        "Specificity":
        specificity

    }





def calibration_metrics(
    y_true,
    y_prob
):

    """
    Calibration evaluation.
    """

    brier = brier_score_loss(
        y_true,
        y_prob
    )


    return {

        "Brier_score":
        brier

    }
