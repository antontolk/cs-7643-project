from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(
        train_path: Path,
        val_path: Path,
        test_path: Path,
        dataset: str = 'MELD',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation and test datasets.

    :param train_path: Path to the train dataset file.
    :type train_path: Path
    :param val_path: Path to the validation dataset file.
    :type val_path: Path
    :param test_path: Path to the test dataset file.
    :type test_path: Path
    :param dataset: The name of the dataset to be loaded.
    :type dataset: str

    :returns: tuple that includes train, validation and test pandas Dataframes.
    :rtype: tuple
    """
    if dataset == 'MELD':

        cols_to_load = [
            'Sr No.', 'Utterance', 'Speaker', 'Emotion', 'Sentiment',
            'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode',
        ]
        dtype: dict = {
            'Sr No.': np.int32,
            'Utterance': str,
            'Speaker': str,
            'Emotion': str,
            'Sentiment': str,
            'Dialogue_ID': np.int32,
            'Utterance_ID': np.int32,
            'Season': np.int32,
            'Episode': np.int32,
        }

        df_train = pd.read_csv(
            train_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )
        df_val = pd.read_csv(
            val_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )
        df_test = pd.read_csv(
            test_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )
    else:
        raise ValueError(f'{dataset} is not supported.')

    return df_train, df_val, df_test
