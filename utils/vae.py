import keras
from keras import layers


class Sampling(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = keras.ops.shape(z_mean)[0]
        dim = keras.ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        sample = z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
        return sample


class Encoder(keras.models.Model):
    def __init__(
        self,
        latent_dim,
        compression_layers,
    ):
        super().__init__()
        # init arguments
        self.latent_dim = latent_dim
        self.compression_layers = compression_layers

        # model building blocks
        self.hidden_layers = []
        for nodes in self.compression_layers:
            self.hidden_layers.append(layers.Dense(nodes, activation=None))
        self.mean_out = layers.Dense(self.latent_dim, activation=None)
        self.log_var_out = layers.Dense(self.latent_dim, activation=None)
        self.sampling = Sampling()

    def call(self, inputs):
        # pass input through all the hidden layers, and finally the output layer
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)

        z_mean = self.mean_out(x)
        z_log_var = self.log_var_out(x)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z


class Decoder(keras.models.Model):
    def __init__(self, decompression_layers, num_classes):
        super().__init__()
        # init arguments
        self.decompression_layers = decompression_layers
        self.num_classes = num_classes

        # model building blocks
        self.hidden_layers = []
        for nodes in self.decompression_layers:
            self.hidden_layers.append(layers.Dense(nodes, activation=None))

        self.decoder_output = []
        for classes in self.num_classes:
            if classes <= 2:
                dec_out = layers.Dense(units=1, activation="sigmoid")
            else:
                dec_out = layers.Dense(units=classes, activation="softmax")
            self.decoder_output.append(dec_out)
        self.concat = layers.Concatenate()

    def call(self, inputs):
        # pass input through all the hidden layers, and finally the separate output layers and concatenate those
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)

        outputs = []
        for layer in self.decoder_output:
            outputs.append(layer(x))
        outputs = self.concat(outputs)

        return outputs


class Predictor(keras.models.Model):

    def __init__(self):
        super().__init__()
        self.predict = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        output = self.predict(inputs)

        return output


class VAE(keras.models.Model):

    def __init__(
        self,
        latent_dim,
        compression_layers,
        decompression_layers,
        num_classes,
    ):
        super().__init__()
        # init arguments
        self.latent_dim = latent_dim
        self.compression_layers = compression_layers
        self.decompression_layers = decompression_layers
        self.num_classes = num_classes

        # model building blocks
        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            compression_layers=self.compression_layers,
        )
        self.decoder = Decoder(
            decompression_layers=self.decompression_layers,
            num_classes=self.num_classes,
        )
        self.predictor = Predictor()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        kl_loss = -0.5 * keras.ops.mean(
            z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        # we add KL loss as regularization term, cross entropy can be added when compiling
        # then the model automatically sums losses together
        reconstructed = self.decoder(z)
        pred = self.predictor(reconstructed)
        return pred


# TBD:
# - extent beyond binary classification
# - add loss factor in KL loss
