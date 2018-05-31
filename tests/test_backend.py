import pytest
import os
import hydro_serving_grpc as hsg
import onnx
from caffe2.proto import caffe2_pb2

import consts
from caffe2_runtime.backend import Caffe2Backend, ONNXBackend, load_serving_backend
from caffe2.python.onnx.frontend import Caffe2Frontend

from tests.downloader import download_googlenet

CAFFE_MODEL_PATH = "tests/models/googlenet_caffe2/files"
ONNX_MODEL_PATH = "tests/models/googlenet_onnx/files"


def prepare_contract(model_path):
    inputs=[
        hsg.ModelField(
            name="data",
            dtype=hsg.DT_FLOAT,
            shape=hsg.TensorShapeProto(
                dim=[
                    hsg.TensorShapeProto.Dim(size=1),
                    hsg.TensorShapeProto.Dim(size=3),
                    hsg.TensorShapeProto.Dim(size=224),
                    hsg.TensorShapeProto.Dim(size=224)
                ]
            )
        )
    ]
    outputs=[
        hsg.ModelField(
            name="prob",
            dtype=hsg.DT_FLOAT
        )
    ]
    sig = hsg.ModelSignature(
        signature_name="infer",
        inputs=inputs,
        outputs=outputs
    )
    contract = hsg.ModelContract(
        signatures=[sig]
    )
    contract_path = os.path.join(model_path, "..", "contract.protobin")
    with open(contract_path, "wb") as file:
        file.write(contract.SerializeToString())
    print(contract)
    return contract


@pytest.fixture()
def download_model():
    print("Downloading model...")
    download_googlenet(CAFFE_MODEL_PATH)
    prepare_contract(CAFFE_MODEL_PATH)


@pytest.fixture()
def convert_model():
    print("Downloading and converting model...")
    download_googlenet(CAFFE_MODEL_PATH)
    prepare_contract(CAFFE_MODEL_PATH)

    predict_net_path = os.path.join(CAFFE_MODEL_PATH, consts.CAFFE2_PREDICT_NET_PB)
    init_net_path = os.path.join(CAFFE_MODEL_PATH, consts.CAFFE2_INIT_NET_PB)

    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_path, 'rb') as f:
        predict_net.ParseFromString(f.read())
        print("Loaded %s", predict_net_path)

    init_net = caffe2_pb2.NetDef()
    with open(init_net_path, 'rb') as f:
        init_net.ParseFromString(f.read())
        print("Loaded %s", init_net_path)

    onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model(
        predict_net=predict_net,
        init_net=init_net,
        value_info={
            "data": (onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        }
    )


def test_caffe2_backend(download_model):
    cb = load_serving_backend(CAFFE_MODEL_PATH)
    assert isinstance(cb, Caffe2Backend)
    result = cb.serve(...)
    assert result is not None


def test_onnx_backend(convert_model):
    cb = load_serving_backend(ONNX_MODEL_PATH)
    assert isinstance(cb, ONNXBackend)
    result = cb.serve(...)
    assert result is not None


    # data_type = onnx.TensorProto.FLOAT
    # data_shape = (1, 3, 224, 224)
    # value_info = {
    #     'data': (data_type, data_shape)
    # }
    value_info = ...

    # predict_net = caffe2_pb2.NetDef()
    # with open(predict_path, 'rb') as f:
    #     predict_net.ParseFromString(f.read())
    #
    # init_net = caffe2_pb2.NetDef()
    # with open(init_path, 'rb') as f:
    #     init_net.ParseFromString(f.read())
    #
    # print(predict_net.external_input)
    # print(predict_net.op[0])
    # onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
    #     predict_net,
    #     init_net,
    #     value_info,
    # )
    #
    # onnx.checker.check_model(onnx_model)
