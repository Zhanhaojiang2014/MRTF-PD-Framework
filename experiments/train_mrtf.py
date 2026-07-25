import os
import sys


sys.path.append("../src")


from utils.config_loader import load_config
from utils.seed import set_seed
from utils.logger import get_logger


from training.train_mrtf import train_mrtf




def main():


    config = load_config(
        "configs/train_config.yaml"
    )


    set_seed(
        config["seed"]
    )


    logger=get_logger(
        "MRTF_training"
    )


    logger.info(
        "Starting MRTF training..."
    )


    logger.info(
        config
    )


    #
    # DataLoader should be created
    # using dataset_split.py
    #


    train_loader=None

    val_loader=None



    model=train_mrtf(

        train_loader,

        val_loader,

        epochs=config["training"]["epochs"]

    )



    logger.info(
        "Training completed."
    )



if __name__=="__main__":

    main()
