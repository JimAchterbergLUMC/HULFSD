from matplotlib import pyplot as plt
import seaborn as sns
from utils import preprocess
import pandas as pd
import numpy as np


def get_column_plots(
    real_data: pd.DataFrame,
    regular_synthetic: pd.DataFrame,
    decoded_synthetic: pd.DataFrame,
    cat_features: list,
    num_features: list,
):
    """
    Get plots of real data, regular synthetic data, and decoded synthetic projections.
    All columns are plotted in a single plot.
    """

    # recode data back to original size (remove one hot encoding)
    real_data = preprocess.decoding_onehot(real_data, cat_features)
    regular_synthetic = preprocess.decoding_onehot(regular_synthetic, cat_features)
    decoded_synthetic = preprocess.decoding_onehot(decoded_synthetic, cat_features)

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
    # plot_dataframe(df=full_df, columns=real_data.columns)

    return plt


def plot_dataframe(df: pd.DataFrame, columns: list, fig: any = [], axs: any = []):
    """
    Loop over columns and plot in a single figure. Histogram for numericals, barplot else.
    """
    for i, col in enumerate(columns):

        dt = preprocess.infer_data_type(df[col])
        if dt == "continuous":
            sns.histplot(data=df, x=col, hue="dataset", alpha=0.5, ax=axs[i])
            # sns.histplot(data=df, x=col, hue="dataset", alpha=0.5)
        else:
            sns.countplot(data=df, x=col, hue="dataset", stat="proportion", ax=axs[i])
            # sns.countplot(data=df, x=col, hue="dataset", stat="proportion")

        # Remove individual legends from subplots
        if i > 0:
            axs[i].get_legend().remove()

    # Extract legend from one subplot and place it outside the subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()


from sklearn.manifold import TSNE


def tsne_projections(real, synthetic):
    X = pd.concat([real, synthetic], axis=0)
    emb = TSNE(
        n_components=2, learning_rate="auto", init="pca", perplexity=35, verbose=10
    ).fit_transform(X)

    labels = np.concatenate(
        (np.zeros((real.shape[0], 1)), np.ones((synthetic.shape[0], 1))), axis=0
    )

    emb = pd.DataFrame(
        np.concatenate((emb, labels), axis=1), columns=["comp1", "comp2", "labels"]
    )

    sns.scatterplot(data=emb, x="comp1", y="comp2", hue="labels", alpha=0.3)
    return plt
