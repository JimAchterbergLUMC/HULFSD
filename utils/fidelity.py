from matplotlib import pyplot as plt
import seaborn as sns
from utils import preprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def get_column_plots(
    real_data: pd.DataFrame,
    regular_synthetic: pd.DataFrame,
    decoded_synthetic: pd.DataFrame,
    cat_features: list,
    num_features: list,
    real_normalizer: any,
    syn_normalizer: any,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot.
    """

    # recode data back to original size (remove one hot encoding)
    print("starting reversing transformations")
    real_data = preprocess.decoding_onehot(real_data, cat_features)
    regular_synthetic = preprocess.decoding_onehot(regular_synthetic, cat_features)
    decoded_synthetic = preprocess.decoding_onehot(decoded_synthetic, cat_features)

    # reverse transform normalization of numerical columns
    # real_data = preprocess.reverse_norm(
    #     data=real_data, fitted_normalizer=real_normalizer, num_features=num_features
    # )
    # regular_synthetic = preprocess.reverse_norm(
    #     data=regular_synthetic,
    #     fitted_normalizer=syn_normalizer,
    #     num_features=num_features,
    # )
    # decoded_synthetic = preprocess.reverse_norm(
    #     data=decoded_synthetic,
    #     fitted_normalizer=real_normalizer,
    #     num_features=num_features,
    # )

    # scale decoded projections to [0,1] for fair comparison of the distribution shape
    decoded_synthetic, _ = preprocess.sklearn_preprocessor(
        processor=MinMaxScaler((0, 1)), data=decoded_synthetic, features=num_features
    )

    print("starting subplot generation")

    # concatenate dataframes with a hue column for easy plotting in same figure
    full_df = pd.concat(
        [
            real_data.assign(dataset="Real"),
            regular_synthetic.assign(dataset="Regular Synthetic"),
            decoded_synthetic.assign(dataset="Decoded Synthetic"),
        ]
    )

    fig, axs = plt.subplots(nrows=len(real_data.columns), ncols=1, figsize=(16, 16))
    plot_dataframe(df=full_df, columns=real_data.columns, fig=fig, axs=axs)

    return plt


def plot_dataframe(df: pd.DataFrame, columns: list, fig: any, axs: any):
    """
    Loop over columns and plot in a single figure. Histogram for numericals, barplot else.
    """
    for i, col in enumerate(columns):
        print(f"currently plotting column {col}")

        dt = preprocess.infer_data_type(df[col])
        if dt == "continuous":
            sns.histplot(data=df, x=col, hue="dataset", alpha=0.5, ax=axs[i])
        else:
            sns.countplot(data=df, x=col, hue="dataset", stat="proportion", ax=axs[i])

        # Remove individual legends from subplots
        if i > 0:
            axs[i].get_legend().remove()

    # Extract legend from one subplot and place it outside the subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
