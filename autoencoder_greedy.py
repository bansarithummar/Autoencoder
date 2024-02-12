import numpy as np
import tensorflow as tf

class AutoencoderLayer:
    def __init__(self, input_dim, hidden_dim, learning_rate):
        self.weights = tf.Variable(tf.random.normal([input_dim, hidden_dim]))
        self.bias_hidden = tf.Variable(tf.zeros([hidden_dim]))
        self.bias_output = tf.Variable(tf.zeros([input_dim]))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def train_step(self, input_data):
        with tf.GradientTape() as tape:
            encoded = tf.nn.sigmoid(tf.matmul(input_data, self.weights) + self.bias_hidden)
            decoded = tf.nn.sigmoid(tf.matmul(encoded, tf.transpose(self.weights)) + self.bias_output)
            loss = tf.reduce_mean(tf.square(input_data - decoded))
        gradients = tape.gradient(loss, [self.weights, self.bias_hidden, self.bias_output])
        self.optimizer.apply_gradients(zip(gradients, [self.weights, self.bias_hidden, self.bias_output]))
        return loss.numpy()

# Sample raw input data
raw_input_data = np.random.rand(1000, 784)  # Example: 1000 samples with 784 features each

# Define hyperparameters
input_dim = raw_input_data.shape[1]
hidden_dims = [256, 128]  # Define the hidden layer dimensions for each layer
learning_rate = 0.001

# Initialize layers
layers = []
for i, hidden_dim in enumerate(hidden_dims):
    if i == 0:
        layer_input_dim = input_dim
    else:
        layer_input_dim = hidden_dims[i-1]
    layers.append(AutoencoderLayer(layer_input_dim, hidden_dim, learning_rate))

# Greedy layer-wise unsupervised pretraining
for layer in layers:
    for epoch in range(10):  # Example: Train each layer for 10 epochs
        loss = layer.train_step(raw_input_data)
        print(f'Layer {layers.index(layer) + 1} - Epoch {epoch + 1}: Loss = {loss:.4f}')
    # Extract features for next layer
    encoded_data = tf.nn.sigmoid(tf.matmul(raw_input_data, layer.weights) + layer.bias_hidden)
    raw_input_data = encoded_data.numpy()

print("Pretraining Complete!")
