import os
import PIL
import random
import keras
from PIL import ImageOps
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import Conv2D
from unet import U_Net

from pets import OxfordPets

base_dir = "/home/kazi/Works/projects/U-Net"

# Dataset Preparation
input_dir = 'images/'
target_dir = 'annotations/trimaps/'

img_size = (160, 160)
num_classes = 3
batch_size = 8

input_img_paths = sorted(
    [
        os.path.join(base_dir, input_dir, fname)
        for fname in os.listdir(os.path.join(base_dir, input_dir))
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(base_dir, target_dir, fname)
        for fname in os.listdir(os.path.join(base_dir, target_dir))
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# print("Number of samples:", len(input_img_paths))
# for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
#     print(input_path, "|", target_path)

# Data Visualization

# Display input image #7
# im = Image.open(input_img_paths[9])
# im.show()

# Display auto-contrast version of corresponding target (per-pixel categories)
# img = ImageOps.autocontrast(Image.open(target_img_paths[9]))
# img.show()


# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# U-Net Model
input_shape = (160, 160, 3)
model = U_Net(input_shape)
# model.summary()

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# # Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)
