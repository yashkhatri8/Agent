#write a sample code to test the helper functions
import os
import sys
import json
import logging
import time         

def setup_logging(log_file):
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")