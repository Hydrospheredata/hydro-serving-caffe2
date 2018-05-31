import os
import logging
from caffe2_runtime.service import create_server
import time

PORT = os.getenv("APP_PORT", "9090")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "NOTSET")
MAX_WORKERS = int(os.getenv("APP_MAX_WORKERS", "10"))

logging.basicConfig(level=LOG_LEVEL)


def run():
    print("hydro-serving Caffe2 runtime")
    server = create_server("/model", MAX_WORKERS, PORT)
    server.start()
    try:
        while True:
            logging.debug("Sleep iteration")
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
    logging.info("Server exited")
