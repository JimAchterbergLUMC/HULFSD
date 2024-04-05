import keras
from keras import layers


def rev_engineer_projections(projections, real, leak_percentage, activation="relu"):
    """
    Function to reverse engineer projections back to original data space.
    Basically that's just taking projections as X, splitting into train/test set by certain percentage (= leaked percentage).
    Here y equals leaked real data.
    """

    return None


def attribute_inference():
    # function to perform an attribute inference attack using synthetic data when sensitive real targets are known
    pass


def distance_to_closest():
    # function to compute DCR metric between synthetic and real data
    pass
