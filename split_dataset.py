import os
import random
import shutil

SOURCE_DIR = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\images"
OUTPUT_DIR = r"C:\Users\prana\Desktop\ISRO IRoc\CV_Final_Dataset\split"

TRAIN_RATIO = 0.8
random.seed(42)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for split, img_list in [("train", train_imgs), ("test", test_imgs)]:
        out_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for img in img_list:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(out_dir, img)
            )

print("Dataset split completed")
