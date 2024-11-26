from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd


def load_dataset(
        train_path: Path,
        val_path: Path,
        test_path: Path,
        dataset: str = 'MELD',
) -> tuple[dict, dict, Any, Any, Any, Any, Any, Any]:
    """Loads train, validation and test datasets.

    :param train_path: Path to the train dataset file.
    :type train_path: Path
    :param val_path: Path to the validation dataset file.
    :type val_path: Path
    :param test_path: Path to the test dataset file.
    :type test_path: Path
    :param dataset: The name of the dataset to be loaded.
    :type dataset: str

    :returns: emotion categories, sentiment categories, tuple of six pandas
      Dataframes that includes train samples and labels, validation samples and
      labels, and test samples and labels.
    :rtype: tuple
    """
    # MELD Dataset
    if dataset == 'MELD':

        cols_to_load = [
            'Sr No.', 'Utterance', 'Speaker', 'Emotion', 'Sentiment',
            # 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode',
        ]
        dtype: dict = {
            'Sr No.': np.int32,
            'Utterance': str,
            'Speaker': str,
            'Emotion': str,
            'Sentiment': str,
            # 'Dialogue_ID': np.int32,
            # 'Utterance_ID': np.int32,
            # 'Season': np.int32,
            # 'Episode': np.int32,
        }

        #######################################################################
        # Train dataset
        df_train = pd.read_csv(
            train_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )
        # Create mapping for emotion and sentiment categories
        emotions = {
            emotion: i
            for i, emotion in enumerate(df_train.loc[:, 'Emotion'].unique())
        }
        sentiments = {
            sentiment: i
            for i, sentiment in enumerate(df_train.loc[:, 'Sentiment'].unique())
        }

        # Replace categories by integers
        df_train['Emotion'] = df_train['Emotion'].map(emotions)
        df_train['Sentiment'] = df_train['Sentiment'].map(sentiments)

        # Split dataset by samples and labels
        df_train_x = df_train.loc[:, ['Utterance', 'Speaker']]
        df_train_y = df_train.loc[:, ['Emotion', 'Sentiment']]

        #######################################################################
        # Validation dataset
        df_val = pd.read_csv(
            val_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )

        # Replace categories by integers
        df_val['Emotion'] = df_val['Emotion'].map(emotions)
        df_val['Sentiment'] = df_val['Sentiment'].map(sentiments)

        # Split dataset by samples and labels
        df_val_x = df_val.loc[:, ['Utterance', 'Speaker']]
        df_val_y = df_val.loc[:, ['Emotion', 'Sentiment']]

        #######################################################################
        # Test dataset
        df_test = pd.read_csv(
            test_path,
            header=0,
            index_col='Sr No.',
            usecols=cols_to_load,
            dtype=dtype,
        )

        # Replace categories by integers
        df_test['Emotion'] = df_test['Emotion'].map(emotions)
        df_test['Sentiment'] = df_test['Sentiment'].map(sentiments)

        # Split dataset by samples and labels
        df_test_x = df_test.loc[:, ['Utterance', 'Speaker']]
        df_test_y = df_test.loc[:, ['Emotion', 'Sentiment']]
    else:
        raise ValueError(f'{dataset} is not supported.')

    return emotions, sentiments, df_train_x, df_train_y, df_val_x, df_val_y, \
        df_test_x, df_test_y
