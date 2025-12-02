import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        self.allocate_buffers()

    def load_engine(self, path):
        with open(path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        for binding in self.engine:
            name = binding
            try:
                shape = self.engine.get_tensor_shape(binding)
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            except AttributeError:
                idx = self.engine.get_binding_index(binding)
                shape = self.engine.get_binding_shape(idx)
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))

            alloc_shape = list(shape)
            if alloc_shape[0] < 0:
                alloc_shape[0] = 1

            size = trt.volume(alloc_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            is_input = False
            try:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
            except AttributeError:
                if self.engine.binding_is_input(binding):
                    is_input = True

            binding_info = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': alloc_shape
            }

            if is_input:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, processed_input):
        input_name = self.inputs[0]['name']
        self.context.set_input_shape(input_name, processed_input.shape)
        np.copyto(self.inputs[0]['host'], processed_input.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'].reshape(out['shape']) for out in self.outputs]


def preprocess_image(image_path, top=0, left=0, height=1000, width=1000):
    # 1. Read original for display later
    original_image = cv2.imread(image_path)

    # Crop
    cropped_image = original_image[top:top + height, left:left + width, :]

    # 2. Prepare for Model
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    image_chw = image_rgb.transpose(2, 0, 1)
    image_norm = image_chw.astype(np.float16) / 255.0
    image_batch = np.expand_dims(image_norm, axis=0)

    return image_batch, cropped_image

def visualize_heatmap(cropped_image,
                      anomaly_map,
                      pred_label,
                      pred_score):
    heatmap = anomaly_map.squeeze()

    heatmap_uint8 = (heatmap* 255).astype(np.uint8)

    h, w = cropped_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    # 4. Apply Colormap (JET is standard for heatmaps: Blue=Cold, Red=Hot)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cropped_image, 0.6, heatmap_color, 0.4, 0)

    # Visualize
    label = "Abnormal" if pred_label == True else "Normal"
    window_name = f"{label}: {pred_score * 100:.2f}%"
    window = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    trt_path = "models/weights/tensorrt/model.trt"
    image_path = "datasets_origin/normal/normal_2.bmp"

    # Load Model
    model = TensorRTInference(trt_path)

    input_batch, cropped_image = preprocess_image(image_path)

    # Inference
    results = model.infer(input_batch)

    # Extract Results
    pred_score = results[0][0][0]
    pred_label = results[1][0][0]
    anomaly_map = results[2][0]

    # Visualize
    visualize_heatmap(cropped_image, anomaly_map, pred_label, pred_score)