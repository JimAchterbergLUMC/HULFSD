from matplotlib import pyplot as plt
import seaborn as sns
from utils import preprocess
import pandas as pd
import numpy as np


def get_column_plots(
    real_data: pd.DataFrame,
    regular_synthetic: pd.DataFrame,
    decoded_synthetic: pd.DataFrame,
    categorical_features: list,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot (histogram for continuous, bars for categoricals)
    """

    # recode data back to original size (remove one hot encoding)
    print("starting reversing onehotting")
    real_data = preprocess.decoding_onehot(real_data, categorical_features)
    regular_synthetic = preprocess.decoding_onehot(
        regular_synthetic, categorical_features
    )
    decoded_synthetic = preprocess.decoding_onehot(
        decoded_synthetic, categorical_features
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

    fig, axs = plt.subplots(nrows=len(real_data.columns), ncols=1, figsize=(8, 16))

    for i, col in enumerate(real_data.columns):
        print(f"currently plotting column {col}")

        dt = preprocess.infer_data_type(full_df[col])
        if dt == "continuous":
            sns.histplot(data=full_df, x=col, hue="dataset", alpha=0.5, ax=axs[i])
        else:
            sns.countplot(
                data=full_df, x=col, hue="dataset", stat="proportion", ax=axs[i]
            )
        # Remove individual legends from subplots
        axs[i].get_legend().remove()

    # Extract legend from one subplot and place it outside the subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    return plt
