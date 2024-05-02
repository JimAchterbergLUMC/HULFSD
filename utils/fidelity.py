from matplotlib import pyplot as plt
import seaborn as sns
from utils import preprocess
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE


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

    # remove feature string from categories to shorten categories for plotting
    def remove_feature_str(cat):
        keep = cat.split("_")[1:]
        return "_".join(keep)

    real_data = real_data.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )
    regular_synthetic = regular_synthetic.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )
    decoded_synthetic = decoded_synthetic.map(
        lambda x: remove_feature_str(x) if isinstance(x, str) else x
    )

    # concatenate dataframes with a hue column for easy plotting in same figure
    full_df = pd.concat(
        [
            real_data.assign(dataset="Real"),
            regular_synthetic.assign(dataset="Regular Synthetic"),
            decoded_synthetic.assign(dataset="Decoded Synthetic"),
        ]
    )

    plot_dataframe(df=full_df, ncols=2, rotate_labels=True)

    return plt


def plot_dataframe(df: pd.DataFrame, ncols: int, rotate_labels: bool):
    """
    Loop over columns and plot in a single figure. Histogram for numericals, barplot else.
    """

    nrows = int(np.ceil(len(df.columns) / ncols))
    fig = plt.figure(figsize=(16, 16))

    for i, col in enumerate([x for x in df.columns if x != "dataset"]):
        ax = plt.subplot(nrows, ncols, i + 1)
        dt = preprocess.infer_data_type(df[col])
        if dt == "continuous":
            sns.histplot(data=df, x=col, hue="dataset", alpha=0.5, ax=ax)
            # sns.histplot(data=df, x=col, hue="dataset", alpha=0.5)
        else:
            sns.countplot(data=df, x=col, hue="dataset", stat="proportion", ax=ax)
            # sns.countplot(data=df, x=col, hue="dataset", stat="proportion")

        # Remove individual legends from subplots
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

        if rotate_labels:
            ax.tick_params(axis="x", labelrotation=45)

    # Extract legend from one subplot and place it outside the subplots
    fig.legend(handles, labels, fontsize=11, loc="upper right")
    fig.suptitle("Feature distributions", fontsize=11)

    plt.tight_layout()


def tsne_projections(real, synthetic):
    X = pd.concat([real, synthetic], axis=0)
    X = X.reset_index(drop=True)
    perplexity = 35
    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        verbose=10,
    ).fit_transform(X)
    emb = pd.DataFrame(tsne, index=np.arange(tsne.shape[0]), columns=["comp1", "comp2"])

    labels = pd.concat(
        [
            pd.Series("Real", index=np.arange(real.shape[0])),
            pd.Series(
                "Synthetic",
                index=np.arange(real.shape[0], synthetic.shape[0] + real.shape[0]),
            ),
        ]
    )

    labels = labels.reset_index(drop=True)
    emb = emb.reset_index(drop=True)
    print(emb)
    print(labels)
    emb = pd.concat([emb, labels], axis=1)
    emb.columns = ["comp1", "comp2", "labels"]

    print(emb)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=emb, x="comp1", y="comp2", hue="labels", alpha=0.1)
    plt.title(f"tSNE plot ({perplexity} perplexity)", fontsize=11)
    plt.tight_layout()
    return plt
