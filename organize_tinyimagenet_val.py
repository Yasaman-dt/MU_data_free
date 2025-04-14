import os
import shutil

data_dir = "/projets/Zdehghani/MU_scar/data/tiny-imagenet-200"
val_dir = os.path.join(data_dir, "val")
val_img_dir = os.path.join(val_dir, "images")
val_annotations = os.path.join(val_dir, "val_annotations.txt")

# Create class subdirectories
with open(val_annotations, "r") as f:
    lines = f.readlines()
    for line in lines:
        parts = line.split("\t")
        img_name = parts[0]
        class_label = parts[1]
        class_dir = os.path.join(val_dir, class_label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move image to corresponding class directory
        src = os.path.join(val_img_dir, img_name)
        dst = os.path.join(class_dir, img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

# Remove the now-empty images directory **only if it exists**
if os.path.exists(val_img_dir):
    shutil.rmtree(val_img_dir)

print("Validation set reorganized successfully!")
