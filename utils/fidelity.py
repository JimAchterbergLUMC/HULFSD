from matplotlib import pyplot as plt
import seaborn as sns
from utils import preprocess
import pandas as pd


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

    # instantiate a subplot figure with correct size
    fig, axs = plt.subplots(nrows=len(real_data.columns), ncols=1, figsize=(10, 8))
    for i, col in enumerate(real_data.columns):
        print(f"plotting feature: {col}")
        for j, (name, df) in enumerate(
            zip(
                ["Real", "Regular Synthetic", "Decoded Synthetic"],
                [real_data, regular_synthetic, decoded_synthetic],
            )
        ):
            dt = preprocess.infer_data_type(real_data[col])
            if dt == "continuous":
                sns.histplot(
                    df[col], ax=axs[i], color=f"C{j}", alpha=0.5, kde=True, label=name
                )
            else:
                sns.barplot(
                    x=col, y=df.index, data=df, ax=axs[i], color=f"C{j}", label=name
                )
            axs[i].set_title(col, fontsize=11)
    # fig.suptitle(fontsize=11)
    plt.legend(fontsize=11)
    plt.tight_layout()

    return plt
