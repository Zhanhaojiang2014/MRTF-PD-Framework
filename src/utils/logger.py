import logging
import os
from datetime import datetime



def get_logger(
    name="MRTF",
    log_dir="logs"
):

    os.makedirs(
        log_dir,
        exist_ok=True
    )


    timestamp=datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    log_file=os.path.join(
        log_dir,
        f"{name}_{timestamp}.log"
    )


    logger=logging.getLogger(
        name
    )


    logger.setLevel(
        logging.INFO
    )


    if not logger.handlers:


        formatter=logging.Formatter(

            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"

        )


        file_handler=logging.FileHandler(
            log_file
        )


        file_handler.setFormatter(
            formatter
        )


        console_handler=logging.StreamHandler()

        console_handler.setFormatter(
            formatter
        )


        logger.addHandler(
            file_handler
        )


        logger.addHandler(
            console_handler
        )


    return logger
