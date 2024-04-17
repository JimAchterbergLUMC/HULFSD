# generate normal synthetic dataset
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import keras
from keras import layers
import os
import numpy as np


# includes scripts for generating synthetic data (regular or projections)


def generate(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    model: str = "copula",
    model_args: dict = {},
    projected: bool = False,
):
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
            metadata=metadata, numerical_distributions=None, default_distribution="beta"
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

    synthesizer.fit(df)
    syn_df = synthesizer.sample(num_rows=sample_size)
    syn_y = syn_df[target]
    syn_X = syn_df.drop(target, axis=1)
    return syn_X, syn_y


def fit_model(
    model: keras.models.Model,
    X: pd.DataFrame,
    y: pd.Series,
    model_args: dict,
    monitor: str = "val_AUC",
):
    """
    Takes a keras model and fits it with checkpointing.
    """

    # setup model checkpointing
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_filepath = os.path.join(model_path, f"{model.name}.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
    )

    model.fit(
        X,
        y,
        batch_size=model_args["batch_size"],
        epochs=model_args["epochs"],
        validation_split=0.3,
        callbacks=[checkpoint_callback],
    )
    model.load_weights(checkpoint_filepath)

    # fits model inplace, does not return anything


def encoder_network(
    input_dim: int,
    latent_size: int,
):
    # build encoder (without bias and activation for easy inversion)
    encoder_input = layers.Input((input_dim,))
    encoder_output = layers.Dense(latent_size, activation=None, use_bias=False)(
        encoder_input
    )
    encoder = keras.models.Model(encoder_input, encoder_output, name="encoder_encoder")

    # build predictor (which predicts from latent space)
    predictor_input = layers.Input((latent_size,))
    predictor_output = layers.Dense(1, activation="sigmoid")(predictor_input)
    predictor = keras.models.Model(
        predictor_input, predictor_output, name="encoder_predictor"
    )

    # build network from encoder and predictor
    network = keras.models.Model(
        encoder_input, predictor(encoder_output), name="encoder_predictor_network"
    )

    # compile the network
    network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

    return encoder, predictor, network


def flip_encoder(encoder: any):
    _, latent_size = encoder.output.shape
    flipped_encoder_input = layers.Input((latent_size,))
    flipped_encoder_output = layers.Dense(latent_size, activation=None)(
        flipped_encoder_input
    )
    flipped_encoder = keras.models.Model(flipped_encoder_input, flipped_encoder_output)

    # find flipped weights and biases and set to layer
    encoder_weights = encoder.layers[1].get_weights()
    flipped_weights = np.squeeze(np.linalg.inv(encoder_weights))
    # we dont use biases
    biases = np.zeros((flipped_weights.shape[0],))
    flipped_dense = flipped_encoder.layers[1]
    flipped_dense.set_weights([flipped_weights, biases])

    return flipped_encoder
