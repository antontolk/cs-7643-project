import logging


def logger_config(logger):
    """Set the logging level"""
    logging.basicConfig(
        filename='logging.log',
        filemode='a',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )

    console_handler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger