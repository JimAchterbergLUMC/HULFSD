# generate normal synthetic dataset
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def generate(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    model: str = "copula",
    model_args: dict = {},
    embedded: bool = False,
):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    target = y.name
    df = pd.concat([X, y], axis=1)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    # for embedded features, all non target features should be detected as numerical
    if embedded:
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
