import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc



def plot_training_curve(
    losses,
    output_path
):


    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        losses
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        "MRTF Training Convergence"
    )


    plt.grid(
        True
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()





def plot_roc_curve(
    y_true,
    y_prob,
    output_path
):


    fpr,tpr,_=roc_curve(
        y_true,
        y_prob
    )


    roc_auc=auc(
        fpr,
        tpr
    )


    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.3f}"
    )


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curve - MRTF"
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()





def plot_model_comparison(
    models,
    accuracy,
    output_path
):


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        models,
        accuracy
    )


    plt.ylabel(
        "Accuracy (%)"
    )


    plt.title(
        "Comparison of MRTF and Baselines"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300
    )


    plt.close()
