import logging


def get_logger(_name):
    # Create a logger
    logger = logging.getLogger(_name)
    logger.setLevel(logging.INFO)  # Set the level on the logger itself

    # Create a handler that outputs to the console
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Set the level on the handler

    # Create a formatter (optional, for nicer output)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    return logger