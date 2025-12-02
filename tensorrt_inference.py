import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time


class TensorRTInference:
    def __init__(self, engine_path, input_shape=(1, 3, 1000, 1000)):
        print(f"Initializing TensorRT Engine from {engine_path}...")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.allocate_buffers()

        # Optimization: Pre-configure context inputs/addresses ONCE
        input_name = self.inputs[0]['name']
        self.context.set_input_shape(input_name, input_shape)

        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        print("Engine initialized successfully.")

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
                mode = self.engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
            except AttributeError:
                idx = self.engine.get_binding_index(binding)
                shape = self.engine.get_binding_shape(idx)
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                is_input = self.engine.binding_is_input(binding)

            alloc_shape = list(shape)
            if alloc_shape[0] < 0: alloc_shape[0] = 1

            size = trt.volume(alloc_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            binding_info = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': alloc_shape,
                'dtype': dtype
            }

            if is_input:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)

    def infer(self, image_raw):
        host_buffer = self.inputs[0]['host']
        input_shape = self.inputs[0]['shape']
        host_view = host_buffer.reshape(input_shape)

        # Normalize and Transpose directly into pinned memory
        host_view[0] = image_raw.transpose(2, 0, 1).astype(np.float16) / 255.0

        # Execute
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return [out['host'].reshape(out['shape']) for out in self.outputs]


def process_pipeline(model, 
                     image_path,
                     top = 0,
                     left = 0,
                     width = 1000,
                     height = 1000):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Crop
    cropped_image = original_image[top:top + height, left:left + width, :]
    cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # inference
    t_start = time.time()
    results = model.infer(cropped_rgb)
    t_cost = (time.time() - t_start) * 1000  # Convert to ms

    # Extract Results
    pred_score = results[0][0][0]
    pred_label = results[1][0][0]
    anomaly_map = results[2][0]

    # Visualize
    visualize_heatmap(cropped_image, anomaly_map, pred_label, pred_score, t_cost, image_path)


def visualize_heatmap(cropped_image, 
                      anomaly_map, 
                      pred_label, 
                      pred_score, 
                      t_cost, 
                      filename):
    heatmap = anomaly_map.squeeze()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    h, w = cropped_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cropped_image, 0.6, heatmap_color, 0.4, 0)

    label_text = "Abnormal" if pred_label else "Normal"
    # Show filename and time in title

    # Add text Overlay on image directly so you don't lose it in window title
    info_text = f"{label_text} ({pred_score * 100:.1f}%) | Time: {t_cost:.2f}ms"
    cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"[{filename}] Inference: {t_cost:.2f} ms")

    cv2.imshow("Result", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    trt_path = "models/weights/tensorrt/model.trt"
    model = TensorRTInference(trt_path)

    image_paths = [
        "datasets_origin/normal/normal.bmp",
        "datasets_origin/abnormal/dop_goc.bmp",
        "datasets_origin/abnormal/khong_khit_goc.bmp",
        "datasets_origin/normal/normal_2.bmp",
        "datasets_origin/normal/normal_3.bmp"
    ]

    for image_path in image_paths:
        process_pipeline(model, image_path)

    cv2.destroyAllWindows()