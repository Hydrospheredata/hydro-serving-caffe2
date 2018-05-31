import logging
import os
from caffe2_runtime import consts
from caffe2.proto import caffe2_pb2
import onnx


class ServingBackend:
    def serve(self, data_dict):
        raise NotImplementedError()


class Caffe2Backend(ServingBackend):
    def __init__(self, predict_net_path, init_net_path):
        self.logger = logging.getLogger("Caffe2Backend")
        self.logger.debug("Loading files...")

        self.predict_net = caffe2_pb2.NetDef()
        with open(predict_net_path, 'rb') as f:
            self.predict_net.ParseFromString(f.read())
            self.logger.debug("Loaded %s", predict_net_path)

        self.init_net = caffe2_pb2.NetDef()
        with open(init_net_path, 'rb') as f:
            self.init_net.ParseFromString(f.read())
            self.logger.debug("Loaded %s", init_net_path)

    def serve(self, data_dict):
        pass


class ONNXBackend(ServingBackend):
    def __init__(self, onnx_path):
        self.logger = logging.getLogger("ONNXBackend")
        self.logger.debug("Loading %s", onnx_path)
        self.model = onnx.load(onnx_path)
        self.logger.debug("Loaded %s graph", self.model.graph.name)

    def serve(self, data_dict):
        pass


def load_serving_backend(model_files_path):
    """
    Detect a model in folder and create appropriate backend for it
    :param model_files_path: path with model files
    :return: instance of a ServingBackend subclass
    """

    logger = logging.getLogger("load_serving_backend")

    files = os.listdir(model_files_path)
    logger.debug("Files in %s: %s", model_files_path, files)

    backend = None

    if consts.CAFFE2_INIT_NET_PB in files and consts.CAFFE2_PREDICT_NET_PB in files:
        logger.info("Caffe2 model detected")
        backend = Caffe2Backend(
            os.path.join(model_files_path, consts.CAFFE2_PREDICT_NET_PB),
            os.path.join(model_files_path, consts.CAFFE2_INIT_NET_PB)
        )
    else:
        onnx_files = list(
            filter(
                lambda x: x == ".onnx",
                map(os.path.splitext, files)
            )
        )
        if onnx_files:
            onnx_file = onnx_files[0]
            logger.info("ONNX model detected (file: %s)", onnx_file)
            backend = ONNXBackend(onnx_file)
        else:
            logger.error("Can't find supported models in %s", model_files_path)

    if backend is None:
        raise RuntimeError("Can't find supported models in {} directory".format(model_files_path))

    return backend
