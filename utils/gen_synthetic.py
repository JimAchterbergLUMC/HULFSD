# generate normal synthetic dataset
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def generate(X, y, model="copula", embedded=False):
    target = y.name
    df = pd.concat([X, y], axis=1)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    # for embedded features, all non target features should be detected as numerical
    if embedded:
        metadata.remove_primary_key()
        for feature in X.columns:
            metadata.update_column(column_name=feature, sdtype="numerical")
    print(metadata)
    if model == "copula":
        synthesizer = GaussianCopulaSynthesizer(
            metadata, numerical_distributions=None, default_distribution="beta"
        )

    synthesizer.fit(df)
    syn_df = synthesizer.sample(num_rows=df.shape[0])
    syn_y = syn_df[target]
    syn_X = syn_df.drop(target, axis=1)
    return syn_X, syn_y
