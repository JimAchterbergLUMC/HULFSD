from sklearn.decomposition import PCA, FactorAnalysis
import keras
from keras import layers
import os
import pandas as pd


def pca(X: pd.DataFrame):
    pca_ = PCA(n_components=X.shape[1])  # retain all components
    embeddings = pca_.fit_transform(X)
    return embeddings


def fa(X: pd.DataFrame):
    fa_ = FactorAnalysis(n_components=X.shape[1])  # retain all components
    embeddings = fa_.fit_transform(X)
    return embeddings


def encoder_decoder(
    X: pd.DataFrame,
    y: pd.Series,
    latent_size: int,
    model_args: dict,
    activation: str = "relu",
):
    # build encoder
    encoder_input = layers.Input((X.shape[1],))
    encoder_x = layers.Dense(2 * latent_size, activation=activation)(encoder_input)
    encoder_x = layers.Dense(int(1.5 * latent_size), activation=activation)(encoder_x)
    encoder_output = layers.Dense(latent_size, activation=activation)(encoder_x)
    encoder = keras.models.Model(encoder_input, encoder_output)

    # build decoder
    decoder_input = layers.Input((latent_size,))
    decoder_output = layers.Dense(1, activation="sigmoid")(decoder_input)
    decoder = keras.models.Model(decoder_input, decoder_output)

    # build network from encoder and decoder
    network = keras.models.Model(encoder_input, decoder(encoder_output))

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

    # fit the network and get the instance with best generalization
    network.fit(
        X,
        y,
        batch_size=model_args["batch_size"],
        epochs=model_args["epochs"],
        validation_split=0.3,
        callbacks=[checkpoint_callback],
    )
    network.load_weights(checkpoint_filepath)

    return encoder, decoder, network
