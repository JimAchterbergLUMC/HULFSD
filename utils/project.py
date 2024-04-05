from sklearn.decomposition import PCA, FactorAnalysis
import keras
from keras import layers
import os
import pandas as pd
import numpy as np
import tensorflow as tf


def pca(X: pd.DataFrame):
    pca_ = PCA(n_components=X.shape[1])  # retain all components
    embeddings = pca_.fit_transform(X)
    return embeddings


def fa(X: pd.DataFrame):
    fa_ = FactorAnalysis(n_components=X.shape[1])  # retain all components
    embeddings = fa_.fit_transform(X)
    return embeddings


def encoder_network(
    X: pd.DataFrame,
    y: pd.Series,
    latent_size: int,
    model_args: dict,
):
    # build encoder
    encoder_input = layers.Input((X.shape[1],))
    # encoder_x = layers.Dense(int(2 * latent_size), activation=layers.LeakyReLU(0.3))(
    #     encoder_input
    # )
    # encoder_x = layers.Dense(int(1.5 * latent_size), activation=layers.LeakyReLU(0.3))(
    #     encoder_x
    # )
    encoder_output = layers.Dense(latent_size, activation=layers.LeakyReLU(0.3))(
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


def flip_encoder(encoder):
    # next step: check whether it really inverts it. also, code can probably be simplified...

    encoder_output_shape = encoder.output_shape
    flipped_encoder_input = layers.Input(encoder_output_shape)
    x = flipped_encoder_input

    # setup list of weights and biases
    flipped_weights = []
    flipped_biases = []

    for encoder_layer in reversed(encoder.layers[1:]):

        # get the inverse of the leaky relu function
        def inverse_leaky_relu(x, alpha):
            return tf.where(x < 0, x / alpha, x)

        # set the correct layer type (infer #units and set inverse leaky relu activation)
        flipped_layer = layers.Dense(
            encoder_layer.units,
            activation=lambda x: inverse_leaky_relu(x, 0.3),
        )

        # add the layer to the model
        x = flipped_layer(x)

        # get the weights and biases
        encoder_weights, encoder_biases = encoder_layer.get_weights()

        # compute the inverse of the weights and biases
        flipped_weights.append(np.linalg.pinv(encoder_weights).T)
        flipped_biases.append(-np.dot(flipped_weights[-1], encoder_biases))
    flipped_encoder_output = x
    flipped_encoder = keras.models.Model(flipped_encoder_input, flipped_encoder_output)
    print(flipped_encoder.summary())

    for i, flipped_layer in enumerate(flipped_encoder.layers[1:]):
        # set the weights and biases to the layer AFTER all layers are added (else weight matrices are incorrect form)
        flipped_layer.set_weights([flipped_weights[i], flipped_biases[i]])

    return flipped_encoder
