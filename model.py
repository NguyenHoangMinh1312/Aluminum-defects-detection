import torch
from anomalib.models import Stfpm
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize, InterpolationMode, Grayscale


class MyModel(Stfpm):
    def __init__(self, layers=["layer2", "layer3"]):
        super().__init__(layers=layers)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.student_model.parameters(),
                                 lr=0.1,
                                 weight_decay=1e-3)

    def configure_pre_processor(self, image_size=None) -> PreProcessor:
        transform = Compose([
            Resize((448, 448), interpolation=InterpolationMode.BILINEAR, antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        ])

        return PreProcessor(transform=transform)

if __name__ == "__main__":
    model = MyModel()
    print(model)

