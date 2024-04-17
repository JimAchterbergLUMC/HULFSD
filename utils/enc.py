import keras
from keras import layers
import numpy as np

# TBD:
# extend beyond binary classification
# extend beyond single layer encoder without bias or activation
# extend beyond unregularized latent space


class Encoder(keras.models.Model):
    def __init__(
        self,
        output_dim,
        input_dim,
        compression_layers,
    ):
        super().__init__()
        # init arguments
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.compression_layers = compression_layers

        # model building blocks
        self.hidden_layers = []
        for nodes in self.compression_layers:
            self.hidden_layers.append(
                layers.Dense(nodes, activation=None, use_bias=False)
            )

        self.out = layers.Dense(self.output_dim, activation=None, use_bias=False)

    def build(self, input_shape):
        # Define the shape of the input to the first hidden layer
        hidden_input_shape = (input_shape[-1],)

        # Initialize the weights of all hidden layers
        for layer in self.hidden_layers:
            layer.build(hidden_input_shape)
            hidden_input_shape = (
                layer.units,
            )  # Update the input shape for the next layer

        # Initialize the weights of the output layer
        self.out.build(hidden_input_shape)

    def call(self, inputs):
        # pass input through all the hidden layers, and finally the output layer
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)

        proj = self.out(x)

        return proj

    def flip(self):
        # set up the flipped model architecture

        rev_compression_layers = self.compression_layers
        if len(rev_compression_layers) > 0:
            rev_compression_layers.reverse()
        flipped_encoder = Encoder(
            output_dim=self.input_dim,
            input_dim=self.output_dim,
            compression_layers=rev_compression_layers,
        )

        # need to build the model before we can set weights
        flipped_encoder.build(input_shape=(self.output_dim,))
        print(flipped_encoder.summary())
        inv_weights = []

        # retrieve weights and invert
        for layer in self.layers:
            if isinstance(layer, layers.Dense):
                weights = layer.get_weights()
                inv = np.squeeze(np.linalg.pinv(weights))
                inv_weights.append(inv)

        # set weights to the corresponding layers in reverse order since layers are reversed
        c = len(inv_weights) - 1
        for layer in flipped_encoder.layers:
            if isinstance(layer, layers.Dense):
                print(layer)
                layer.set_weights(weights=[inv_weights[c]])
                c -= 1

        return flipped_encoder


class Predictor(keras.models.Model):

    def __init__(self):
        super().__init__()
        self.predict = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        output = self.predict(inputs)

        return output


class EncoderModel(keras.models.Model):

    def __init__(self, latent_dim, input_dim, compression_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_layers = compression_layers
        self.input_dim = input_dim

        self.encoder = Encoder(
            output_dim=self.latent_dim,
            input_dim=self.input_dim,
            compression_layers=self.compression_layers,
        )
        self.predictor = Predictor()

    def call(self, inputs):
        proj = self.encoder(inputs)
        pred = self.predictor(proj)
        return pred
