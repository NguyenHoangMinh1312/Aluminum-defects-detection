import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from lightning.pytorch.callbacks import TQDMProgressBar

from model import MyModel

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    datamodule = Folder(
        name="aluminum-defects",
        root="./1000x1000_augmented_datasets",
        normal_dir="normal",
        abnormal_dir="abnormal",
        mask_dir="mask",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
    )
    datamodule.setup()

    model = MyModel.load_from_checkpoint('results/MyModel/aluminum-defects/v0/weights/lightning/model.ckpt')

    engine = Engine(
        max_epochs=200,
        devices=1,
        accelerator="auto",
        default_root_dir="results",
        callbacks=[TQDMProgressBar(refresh_rate=10)]
    )

    engine.fit(model=model, datamodule=datamodule)

    # Test after training
    test_results = engine.test(model=model, datamodule=datamodule)
    print(test_results)