import sys

sys.path.append("../src")


from utils.config_loader import load_config
from utils.seed import set_seed
from utils.logger import get_logger




def build_variant(
    name,
    config
):


    model_config=config["experiments"][name]


    print(
        f"""
Running Ablation:

{name}

CAFT:
{model_config['CAFT']}

RASL:
{model_config['RASL']}

XAI-ITL:
{model_config['XAI_ITL']}

"""
    )


    return model_config




def main():


    config=load_config(

        "configs/ablation_config.yaml"

    )


    set_seed(
        config["training"]["seed"]
    )


    logger=get_logger(
        "MRTF_ablation"
    )



    for experiment in config["experiments"]:


        logger.info(

            f"Starting {experiment}"

        )


        variant=build_variant(

            experiment,

            config

        )


        #
        # Train modified MRTF
        #
        # Results saved here
        #



        logger.info(

            f"{experiment} completed"

        )





if __name__=="__main__":

    main()
