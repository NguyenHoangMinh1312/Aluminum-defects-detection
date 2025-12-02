import os
import cv2
import numpy as np
from anomalib.deploy import TorchInferencer

os.environ["TRUST_REMOTE_CODE"] = "1"

def preprocess_image(image_path,
                     top = 0,
                     left = 0,
                     height = 1000,
                     width = 1000,
                     ):
    image = cv2.imread(image_path)
    image = image[top:top+height, left:left+width, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# inference on .pt model
def torch_inferencer(pt_path, image_path):
    # load model
    model = TorchInferencer(pt_path)

    # preprocess (crop) the image
    input_image = preprocess_image(image_path)
    h, w = input_image.shape[:2]

    # inference
    prediction = model.predict(input_image)

    # Anomaly map
    anomaly_map = prediction.anomaly_map.detach().cpu().numpy()
    anomaly_map = anomaly_map.squeeze()
    anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Apply colormap (expects single channel input)
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

    # Overlay the processed image with heatmap
    alpha = 0.5
    overlay = cv2.addWeighted(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                              1 - alpha,
                              heatmap,
                              alpha,
                              0)

    # Visualize the heatmap
    pred_score = prediction.pred_score.detach().cpu().numpy()[0][0]
    window_name = f"Heatmap: {pred_score * 100:.2f}%"
    window = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pt_path = "models/weights/torch/model.pt"
    image_path = "datasets_origin/normal/normal.bmp"

    torch_inferencer(pt_path, image_path)










