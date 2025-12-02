from anomalib.deploy import ExportType
from anomalib.engine import Engine

from model import MyModel


def toTorch(ckpt_path):
    ckpt_model = MyModel.load_from_checkpoint(ckpt_path)
    engine = Engine()
    engine.export(
        model=ckpt_model,
        export_type=ExportType.TORCH,
        export_root="models",
        model_file_name="model",
    )
    
def toONNX(ckpt_path, input_size=(1000, 1000)):
    ckpt_model = MyModel.load_from_checkpoint(ckpt_path)
    engine = Engine()

    engine.export(
        model=ckpt_model,
        export_type=ExportType.ONNX,
        export_root="models",
        model_file_name="model",
        ckpt_path = ckpt_path,
        input_size=input_size,
        onnx_kwargs= {
            "dynamo": False
        }
    )
if __name__ == "__main__":
    ckpt_path = "results/MyModel/aluminum-defects/v0/weights/lightning/model.ckpt"
    toONNX(ckpt_path)
    toTorch(ckpt_path)


