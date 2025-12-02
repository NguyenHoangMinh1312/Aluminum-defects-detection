trtexec --onnx="/home/hoang-minh-nguyen/Documents/INNOVISION/aluminum-angles/models/weights/onnx/model.onnx" --saveEngine="/home/hoang-minh-nguyen/Documents/INNOVISION/aluminum-angles/models/weights/tensorrt/model.trt" --fp16 --minShapes=input:1x3x1000x1000 --optShapes=input:1x3x1000x1000 --maxShapes=input:16x3x1000x1000

