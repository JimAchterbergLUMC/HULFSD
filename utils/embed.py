from sklearn.decomposition import PCA, FactorAnalysis
import keras
from keras import layers
import os


def pca(X):
    pca_ = PCA(n_components=X.shape[1])  # retain all components
    embeddings = pca_.fit_transform(X)
    return embeddings


def fa(X):
    fa_ = FactorAnalysis(n_components=X.shape[1])  # retain all components
    embeddings = fa_.fit_transform(X)
    return embeddings


def autoencoder(X, y, latent_size, model_args, activation="relu"):
    # build encoder
    encoder_input = layers.Input((X.shape[1]))
    x = layers.Dense(2 * latent_size, activation=activation)(encoder_input)
    encoder_output = layers.Dense(latent_size, activation=activation)(x)
    encoder = keras.models.Model(encoder_input, encoder_output)

    # build decoder
    decoder_input = layers.Input((latent_size))
    decoder_output = layers.Dense(1, activation="sigmoid")(decoder_input)
    decoder = keras.models.Model(decoder_input, decoder_output)

    # build autoencoder from encoder and decoder
    ae = keras.models.Model(encoder_input, decoder(encoder_output))

    # compile the autoencoder
    ae.compile(optimizer="adam", loss="binary_crossentropy", metrics="auc")

    # fit the autoencoder (with early stopping)
    if not os.path.exists("models"):
        os.makedirs("models")
    filepath = "models/ae.keras"
    early_stopping = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    )
    ae.fit(
        X,
        y,
        batch_size=model_args["batch_size"],
        epochs=model_args["epochs"],
        callbacks=early_stopping,
    )
    ae.load_weights(early_stopping)

    # get the embeddings
    embeddings = encoder.predict(X)

    # get the output for evaluation
    predictions = ae.predict(X)

    return (
        predictions,
        embeddings,
    )  # encoder, decoder, ae
