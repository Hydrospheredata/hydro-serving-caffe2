import hydro_serving_grpc as hsg
import grpc
import os
from concurrent import futures
import logging
import uuid
from caffe2_runtime.backend import load_serving_backend
from caffe2_runtime import consts


class Caffe2Service(hsg.PredictionServiceServicer):
    def __init__(self, model_path):
        self.logger = logging.getLogger("Caffe2Service")
        self.logger.debug("Initializing Caffe2Service (model_path=%s)", model_path)
        self.model_path = model_path
        self.contract = self._load_contract(os.path.join(model_path, consts.CONTRACT_FILE))
        self.model = load_serving_backend(os.path.join(model_path, consts.FILES_FOLDER))

    def Predict(self, request, context):
        req_id = uuid.uuid4()
        self.logger.info("Incoming request %s \n Data: %s \n Context: %s", req_id, request, context)

        processed_request = self.preprocess(request)
        result = self.model.serve(processed_request)
        processed_result = self.postprocess(result)

        self.logger.info("Response %s \n Data: %s", req_id, processed_result)
        return processed_result

    def preprocess(self, request):
        raise NotImplementedError()

    def postprocess(self, response):
        raise NotImplementedError()

    def _load_contract(self, contract_path):
        self.logger.debug("Loading contract from %s", contract_path)
        contract = hsg.ModelContract()
        with open(contract_path, "rb") as f:
            contract.ParseFromString(f.read())
        self.logger.debug("Contract loaded")
        return contract


def create_server(model_path, max_workers, port):
    logger = logging.getLogger("server-factory")
    logger.info("Initializing Caffe2 runtime.")
    service = Caffe2Service(model_path)
    logger.debug("Caffe2Service is created")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    logger.debug("GRPC server is created with max_workers={}".format(max_workers))
    hsg.add_PredictionServiceServicer_to_server(service, server)

    full_port = "[::]:" + port
    server.add_insecure_port(full_port)
    logger.debug("GRPC server is assigned to port {}".format(full_port))
    return server
