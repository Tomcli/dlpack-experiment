import torch
import tensorflow as tf
from torch.utils import dlpack
import os


for i in range(5):
    # -------------------------------
    # Step 1: Create a PyTorch Tensor
    # -------------------------------
    # Simulating a large input tensor with shape [Batch=204762, Features=208]
    pytorch_tensor = torch.randn(204762, 208, device="cuda")  # Very large tensor on GPU
    print("PyTorch Tensor Shape:", pytorch_tensor.shape)

    # -------------------------------
    # Step 2: Export PyTorch Tensor as DLPack
    # -------------------------------
    dlpack_capsule = dlpack.to_dlpack(pytorch_tensor)

    # -------------------------------
    # Step 3: Import DLPack into TensorFlow
    # -------------------------------
    with tf.device('/GPU:' + str(os.environ["LOCAL_RANK"])):  # Ensure GPU compatibility
        tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_capsule)
        
    print("TensorFlow Tensor Shape:", tf_tensor.shape)

    # -------------------------------
    # Step 4: Build a Dense TensorFlow Model for Tabular-like Data
    # -------------------------------
    class DenseModel(tf.keras.Model):
        def __init__(self):
            super(DenseModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(512, activation='relu')  # First hidden layer
            self.dense2 = tf.keras.layers.Dense(256, activation='relu')  # Second hidden layer
            self.dense3 = tf.keras.layers.Dense(128, activation='relu')  # Third hidden layer
            self.dropout = tf.keras.layers.Dropout(0.4)  # Regularization
            self.output_layer = tf.keras.layers.Dense(10, activation='softmax')  # 10-class output

        def call(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dropout(x)
            x = self.dense3(x)
            return self.output_layer(x)

    # Instantiate the model
    model = DenseModel()

    # -------------------------------
    # Step 5: Feed Tensor into TensorFlow Model
    # -------------------------------
    with tf.device('/GPU:' + str(os.environ["LOCAL_RANK"])):  # Ensure model and data are on the GPU
        output = model(tf_tensor)

    print("Model Output Shape:", output.shape)  # Expected shape: (204762, 10)
    print("Model Output:", output)
