# generate normal synthetic dataset
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.decomposition import PCA, FactorAnalysis
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
            metadata, numerical_distributions=None, default_distribution="beta"
        )

    synthesizer.fit(df)
    syn_df = synthesizer.sample(num_rows=sample_size)
    syn_y = syn_df[target]
    syn_X = syn_df.drop(target, axis=1)
    return syn_X, syn_y


def encoder_network(
    X: pd.DataFrame,
    y: pd.Series,
    latent_size: int,
    model_args: dict,
):
    # build encoder (without bias and activation for easy inversion)
    encoder_input = layers.Input((X.shape[1],))
    encoder_output = layers.Dense(latent_size, activation=None, use_bias=False)(
        encoder_input
    )
    encoder = keras.models.Model(encoder_input, encoder_output)

    # build predictor (which predicts from latent space)
    predictor_input = layers.Input((latent_size,))
    predictor_output = layers.Dense(1, activation="sigmoid")(predictor_input)
    predictor = keras.models.Model(predictor_input, predictor_output)

    # build network from encoder and predictor
    network = keras.models.Model(encoder_input, predictor(encoder_output))

    # compile the network
    network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

    # setup model checkpointing
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_filepath = os.path.join(model_path, "network.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_AUC",
        save_best_only=True,
        save_weights_only=True,
    )

    # fit the network and get the instance with best generalization on a validation set
    network.fit(
        X,
        y,
        batch_size=model_args["batch_size"],
        epochs=model_args["epochs"],
        validation_split=0.3,
        callbacks=[checkpoint_callback],
    )
    network.load_weights(checkpoint_filepath)

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


def pca(X: pd.DataFrame):
    pca_ = PCA(n_components=X.shape[1])  # retain all components
    embeddings = pca_.fit_transform(X)
    return embeddings


def fa(X: pd.DataFrame):
    fa_ = FactorAnalysis(n_components=X.shape[1])  # retain all components
    embeddings = fa_.fit_transform(X)
    return embeddings
