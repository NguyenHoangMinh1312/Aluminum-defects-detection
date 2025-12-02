import glob
import os
import random

import cv2
from tqdm import tqdm


def apply_augmentation(image):
    augmented = image.copy()
    (h, w) = augmented.shape[:2]

    if random.random() > 0.5:
        center = (w // 2, h // 2)
        # Random angle between -5 and 5
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # CRITICAL FIX: Use BORDER_REFLECT to avoid black artifacts
        augmented = cv2.warpAffine(
            augmented, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )

    # 2. Brightness/Contrast (Subtle)
    if random.random() > 0.5:
        # Tighter constraints: 0.9-1.1 contrast, ±20 brightness
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    return augmented

def isAbnormalImage(image_path):
    components = image_path.split("/")
    for component in components:
        if component == "abnormal":
            return True
    return False


def data_augmentation(datasets_origin_dir,
                      normal_augment_times = 100,
                      abnormal_augment_times = 10,
                      top = 0,
                      left = 0,
                      width = 1000,
                      height = 1000):
    if not os.path.exists(datasets_origin_dir):
        print("Dataset does not exist")
        return

    image_paths = glob.glob(os.path.join(datasets_origin_dir, "**", "*.bmp"), recursive=True)
    progress_bar = tqdm(image_paths,  desc= "Data augmentation progress", colour = "green")

    for imagePath in progress_bar:
        try:
            base_image = cv2.imread(imagePath)
            base_image = base_image[top:top+height, left:left+width, :]
        except Exception as e:
            print(f"Skipping {os.path.basename(imagePath)}: {e}")
            continue

        rel_path = os.path.relpath(os.path.dirname(imagePath), datasets_origin_dir)
        save_dir = os.path.join("1000x1000_augmented_datasets", rel_path)

        os.makedirs(save_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(imagePath))[0]

        # Save the un-augmented preprocessed version first (important!)
        cv2.imwrite(os.path.join(save_dir, f"{image_name}.png"), base_image)

        cnt = abnormal_augment_times if isAbnormalImage(imagePath) else normal_augment_times

        # Generate augmentations
        for i in range(cnt):
            aug_img = apply_augmentation(base_image)
            save_path = os.path.join(save_dir, f"{image_name}_aug_{i}.png")
            cv2.imwrite(save_path, aug_img)

    print("Augmentation complete.")

if __name__ == '__main__':
    data_augmentation("datasets_origin")