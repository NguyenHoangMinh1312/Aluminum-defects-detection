trtexec --onnx="models/weights/onnx/model.onnx" --saveEngine="models/weights/tensorrt/model.trt" --fp8 --minShapes=input:1x3x1000x1000 --optShapes=input:1x3x1000x1000 --maxShapes=input:32x3x1000x1000

