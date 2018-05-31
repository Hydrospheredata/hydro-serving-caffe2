from caffe2_runtime.service import create_server
import logging


def test_server_creation():
    logging.getLogger().setLevel(logging.DEBUG)
    server = create_server("tests/models/googlenet", 10, "8080")
    assert server is not None
    print(server)
    server.start()
    server.stop(100)
