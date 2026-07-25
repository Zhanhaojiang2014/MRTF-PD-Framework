import yaml
import os



def load_config(
    config_path
):


    """
    Load YAML configuration file.
    """


    if not os.path.exists(
        config_path
    ):

        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )



    with open(
        config_path,
        "r"
    ) as file:


        config=yaml.safe_load(
            file
        )


    return config





def merge_configs(
    *configs
):


    """
    Merge multiple YAML dictionaries.
    """


    merged={}


    for cfg in configs:

        merged.update(
            cfg
        )


    return merged
