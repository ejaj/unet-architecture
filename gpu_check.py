import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.sysconfig.get_build_info())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
