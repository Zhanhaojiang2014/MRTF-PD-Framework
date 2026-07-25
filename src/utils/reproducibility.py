import sys
import torch
import platform
import json
import os



def save_environment_info(
    output_file="environment_info.json"
):


    info={

        "python_version":
        sys.version,


        "platform":
        platform.platform(),


        "pytorch_version":
        torch.__version__,


        "cuda_available":
        torch.cuda.is_available()

    }



    if torch.cuda.is_available():

        info["gpu"] = torch.cuda.get_device_name(
            0
        )


        info["cuda_version"] = torch.version.cuda



    with open(
        output_file,
        "w"
    ) as f:


        json.dump(
            info,
            f,
            indent=4
        )



    return info





def check_reproducibility():

    """
    Check deterministic settings.
    """


    status={

        "cudnn_deterministic":
        torch.backends.cudnn.deterministic,


        "cudnn_benchmark":
        torch.backends.cudnn.benchmark

    }


    return status





def create_experiment_directory(
    root="experiments"
):


    os.makedirs(
        root,
        exist_ok=True
    )


    return root
