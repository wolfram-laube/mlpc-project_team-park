import tensorflow as tf
from tensorflow.keras import layers

class KolmogorovArnoldNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_univariate_functions, hidden_units, num_classes, **kwargs):
        super(KolmogorovArnoldNetwork, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_univariate_functions = num_univariate_functions
        self.hidden_units = hidden_units
        self.num_classes = num_classes

        # Univariate functions layers
        self.univariate_layers = [layers.Dense(hidden_units, activation='tanh') for _ in range(num_univariate_functions)]

        # Combination layer
        self.combination_layer = layers.Dense(hidden_units, activation='relu')

        # Output layer
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply univariate functions
        univariate_outputs = [layer(inputs) for layer in self.univariate_layers]

        # Combine outputs
        combined_output = tf.concat(univariate_outputs, axis=-1)
        combined_output = self.combination_layer(combined_output)

        # Flatten and apply final output layer
        combined_output = tf.reduce_mean(combined_output, axis=2)  # Average over feature dimension
        return self.output_layer(combined_output)

    def get_config(self):
        config = super(KolmogorovArnoldNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_univariate_functions': self.num_univariate_functions,
            'hidden_units': self.hidden_units,
            'num_classes': self.num_classes,
        })
        return config
