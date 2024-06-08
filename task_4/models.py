# models.py
import tensorflow as tf

class KolmogorovArnoldNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_univariate_functions, hidden_units, num_classes, **kwargs):
        super(KolmogorovArnoldNetwork, self).__init__(**kwargs)
        self.num_univariate_functions = num_univariate_functions
        self.hidden_units = hidden_units

        # Univariate functions layers
        self.univariate_layers = [tf.keras.layers.Dense(hidden_units, activation='tanh') for _ in range(num_univariate_functions)]

        # Combination layer
        self.combination_layer = tf.keras.layers.Dense(hidden_units, activation='relu')

        # Output layer
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply univariate functions
        univariate_outputs = [layer(inputs) for layer in self.univariate_layers]

        # Combine outputs
        combined_output = tf.concat(univariate_outputs, axis=-1)
        combined_output = self.combination_layer(combined_output)

        # Flatten and apply final output layer
        combined_output = tf.reduce_mean(combined_output, axis=2)  # Average over feature dimension
        return self.output_layer(combined_output)
