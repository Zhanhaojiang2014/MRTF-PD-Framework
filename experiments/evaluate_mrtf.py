import sys

sys.path.append("../src")



from utils.config_loader import load_config
from utils.logger import get_logger


from evaluation.metrics import (
    classification_metrics
)


from evaluation.generate_results import (
    generate_table_results
)



import torch



def main():


    config=load_config(
        "configs/eval_config.yaml"
    )


    logger=get_logger(
        "MRTF_evaluation"
    )


    logger.info(
        "Loading MRTF checkpoint..."
    )



    checkpoint_path=config["checkpoint"]["path"]


    if not torch.cuda.is_available():

        device="cpu"

    else:

        device="cuda"



    logger.info(
        f"Device: {device}"
    )



    #
    # Load model
    #


    predictions={}



    #
    # Test inference
    #
    # predictions format:
    #
    # {
    # "MRTF":
    # {
    # y_true,
    # y_pred,
    # y_prob
    # }
    # }



    results=generate_table_results(

        predictions,

        "results/mrtf_results.csv"

    )



    logger.info(
        results
    )



if __name__=="__main__":

    main()
