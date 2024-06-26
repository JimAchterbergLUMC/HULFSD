import numpy as np
import pandas as pd
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
    CTGANSynthesizer,
)
from sdv.metadata import SingleTableMetadata
import os
import keras
from keras import layers
from sklearn.utils.class_weight import compute_class_weight


# includes scripts for generating synthetic data (regular or projections)


def inverse_leaky_relu(x):
    if x >= 0:
        return x
    else:
        return x / 0.01


def inverse_sigmoid(x):
    return keras.ops.log(x / (1 - x))


def inverse_pass(x):
    return x


def sdv_generate(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    model: str = "copula",
    model_args: dict = {},
    projected: bool = False,
):
    """
    Generates tabular synthetic data using a specified model from the Synthetic Data Vault.
    """
    X = X.copy()
    y = y.copy()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    target = y.name
    df = pd.concat([X, y], axis=1)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    # for embedded features, all non target features should be detected as numerical
    if projected:
        metadata.remove_primary_key()
        for feature in X.columns:
            metadata.update_column(column_name=feature, sdtype="numerical")
    if model == "copula":
        synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            numerical_distributions=None,
            default_distribution="beta",
        )
    elif model == "vae":
        synthesizer = TVAESynthesizer(
            metadata=metadata,
            cuda=False,
            compress_dims=(128, 128),
            embedding_dim=128,
            decompress_dims=(128, 128),
            epochs=300,
            batch_size=500,
            loss_factor=2,
            l2scale=1e-5,
        )

    elif model == "gan":
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            batch_size=500,
            verbose=False,
            epochs=300,
            cuda=False,
            pac=10,
        )

    synthesizer.fit(df)
    syn_df = synthesizer.sample(num_rows=sample_size)
    syn_y = syn_df[target]
    syn_X = syn_df.drop(target, axis=1)
    return syn_X, syn_y


def fit_model(
    model: keras.models.Model,
    X: pd.DataFrame,
    y: pd.Series,
    loss: str,
    model_args: dict,
    metric: str,
):
    """
    Takes a keras model and fits it with checkpointing to retrieve the best generalizing model seen during training.
    """

    X = X.copy()
    y = y.copy()
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    # first compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[metric],
    )

    # setup model checkpointing
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_filepath = os.path.join(model_path, f"{model.name}.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_" + metric,
        save_best_only=True,
        save_weights_only=True,
    )

    model.fit(
        X,
        y,
        validation_split=0.3,
        callbacks=[checkpoint_callback],
        **model_args,
    )

    model.load_weights(checkpoint_filepath)

    # fits model inplace, does not return anything


# TBD:
# extend beyond binary classification


class Encoder(keras.models.Model):
    def __init__(
        self,
        output_dim,
        input_dim,
        compression_layers=[],
        activation="sigmoid",
        reverse=False,
    ):
        super().__init__()
        # init arguments
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.reverse = reverse
        self.compression_layers = compression_layers
        self.activation = activation

        # model building blocks
        self.hidden_layers = []
        for nodes in self.compression_layers:
            self.hidden_layers.append(
                layers.Dense(nodes, activation=None, use_bias=False)
            )

        if reverse:
            self.activation = None

        self.out = layers.Dense(
            self.output_dim,
            activation=self.activation,
            use_bias=False,
        )

        if self.activation == "sigmoid":
            inv_func = inverse_sigmoid
        elif self.activation == "leaky_relu":
            inv_func = inverse_leaky_relu
        else:
            inv_func = inverse_pass

        self.inverse_activation = layers.Lambda(inv_func)

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

        if self.reverse:
            x = self.inverse_activation(x)

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
            reverse=True,
        )

        # need to build the model before we can set weights
        flipped_encoder.build(input_shape=(self.output_dim,))
        inv_weights = []
        # biases = []

        # retrieve weights and invert
        for layer in self.layers:
            if isinstance(layer, layers.Dense):
                # weights, bias = layer.get_weights()
                weights = layer.get_weights()  # in case we dont use bias nodes
                inv = np.squeeze(np.linalg.inv(weights))
                inv_weights.append(inv)
                # bias = -np.dot(bias.T, inv).T
                # biases.append(bias)

        # set weights to the corresponding layers in reverse order since layers are reversed
        c = len(inv_weights) - 1
        for layer in flipped_encoder.layers:
            if isinstance(layer, layers.Dense):
                # layer.set_weights(weights=[inv_weights[c], biases[c]])
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

    def __init__(
        self,
        latent_dim,
        input_dim,
        compression_layers,
        activation="sigmoid",
        name="EncoderModel",
    ):
        super().__init__(
            name=name,
        )
        self.latent_dim = latent_dim
        self.compression_layers = compression_layers
        self.input_dim = input_dim
        self.activation = activation

        self.encoder = Encoder(
            output_dim=self.latent_dim,
            input_dim=self.input_dim,
            compression_layers=self.compression_layers,
            activation=self.activation,
        )
        self.predictor = Predictor()

    def call(self, inputs):
        proj = self.encoder(inputs)
        pred = self.predictor(proj)
        return pred


class Decoder(keras.models.Model):

    def __init__(self, output_dim):
        super().__init__()
        # self.dense = layers.Dense(units=output_dim)
        self.out = layers.Dense(units=output_dim, activation="sigmoid")

    def call(self, inputs):
        return self.out(inputs)


class Keras_binary_MCC(keras.metrics.Metric):

    def __init__(self, name="MCC"):
        super().__init__(name=name)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # get predictions and values as booleans
        y_pred = keras.ops.round(y_pred)
        y_true = keras.ops.cast(y_true, "bool")
        y_pred = keras.ops.cast(y_pred, "bool")

        # find values for tp, tn, fp,fn
        tp_vals = keras.ops.logical_and(
            keras.ops.equal(y_true, True), keras.ops.equal(y_pred, True)
        )
        self.tp.assign_add(keras.ops.sum(tp_vals))

        tn_vals = keras.ops.logical_and(
            keras.ops.equal(y_true, False), keras.ops.equal(y_pred, False)
        )
        self.tn.assign_add(keras.ops.sum(tn_vals))

        fp_vals = keras.ops.logical_and(
            keras.ops.equal(y_true, False), keras.ops.equal(y_pred, True)
        )
        self.fp.assign_add(keras.ops.sum(fp_vals))

        fn_vals = keras.ops.logical_and(
            keras.ops.equal(y_true, True), keras.ops.equal(y_pred, False)
        )
        self.fn.assign_add(keras.ops.sum(fn_vals))

    def result(self):
        num = (self.tp * self.tn) - (self.fp * self.fn)
        denom = keras.ops.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        result = num / denom
        return result

    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
