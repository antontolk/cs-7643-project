"""Data processing realted scripts"""
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.tokenizer_bpe import TokenizerBPE
from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)

def meld_processing(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        labels: list,
        encode_speakers: bool,
        utterance_processing: str,
        ngram: tuple,
        batch_size: int = 32,
        shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Preprocessing MELD dataset

    :param df_train: Training datasaet.
    :type df_train: pandas.DataFrame
    :param df_val: Validation datasaet.
    :type df_val: pandas.DataFrame
    :param df_test: Test datasaet.
    :type df_test: pandas.DataFrame
    :param labels: label columns in DataFrames to be extracted.
    :type labels: list
    :param encode_speakers: If True, the Speaker column will be added to
        samples as One-Hot vectors.
    :param utterance_processing: Options:
        ngram - if ngram is (1, 1) Bag of Word, else N-Grams
        BPE - Byte Pair Encoding
    :type utterance_processing: str
    :param ngram: The lower and upper boundary of the range of n-values for
        different word n-grams or char n-grams to be extracted.
    :type ngram: tuple
    :param batch_size: The batch size
    :type batch_size: int
    :param shuffle: If True, datasets will be shuffled
    :type shuffle: bool

    :return: Train, Val and Test dataloaders and saved categories
    :rtype: tuple[DataLoader, DataLoader, DataLoader, dict]
    """
    categories = {}

    logger.info('Data preprocessing has been started')

    #######################################################################
    # Replace categories by integers
    # Emotions
    emotions = {
        emotion: i
        for i, emotion in enumerate(df_train.loc[:, 'Emotion'].unique())
    }
    categories['emotions'] = list(emotions.keys())
    logger.info('Emotions categories: %s', emotions)

    df_train['Emotion'] = df_train['Emotion'].map(emotions)
    df_val['Emotion'] = df_val['Emotion'].map(emotions)
    df_test['Emotion'] = df_test['Emotion'].map(emotions)
    logger.info('Emotion categories have been replaced')

    # Sentiments
    sentiments = {
        sentiment: i
        for i, sentiment in enumerate(df_train.loc[:, 'Sentiment'].unique())
    }
    categories['sentiments'] = list(sentiments.keys())
    logger.info('Sentiments categories: %s', sentiments)

    df_train['Sentiment'] = df_train['Sentiment'].map(sentiments)
    df_val['Sentiment'] = df_val['Sentiment'].map(sentiments)
    df_test['Sentiment'] = df_test['Sentiment'].map(sentiments)
    logger.info('Sentiments categories have been replaced')

    #######################################################################
    # Transformations of the Utterance column
    if utterance_processing == 'counts':
        logger.info(
            'Utterance will be transformed using CountVectorizer, N-Grams: %s',
            ngram,
        )
        count_vect = CountVectorizer(
            lowercase=True,
            ngram_range=ngram,
        )
        count_vect.fit(df_train.loc[:, 'Utterance'])
        X_train = count_vect.transform(df_train.loc[:, 'Utterance']).toarray()
        X_val = count_vect.transform(df_val.loc[:, 'Utterance']).toarray()
        X_test = count_vect.transform(df_test.loc[:, 'Utterance']).toarray()
        categories['utterance'] = count_vect.get_feature_names_out()
        logger.info(
            'Utterance have be transformed using CountVectorizer.Train %s.'
            'Val: %s. Test: %s',
            X_train.shape, X_val.shape, X_test.shape,
        )

        # Normalise count values
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        categories['scaler'] = scaler.get_feature_names_out()
        logger.info('Count values have been normalised.')

    # TF-IDF transformation
    elif utterance_processing == 'TF-IDF':
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram,
            smooth_idf=True,
        )
        vectorizer.fit(df_train.loc[:, 'Utterance'])
        X_train = vectorizer.transform(df_train.loc[:, 'Utterance']).toarray()
        X_val = vectorizer.transform(df_val.loc[:, 'Utterance']).toarray()
        X_test = vectorizer.transform(df_test.loc[:, 'Utterance']).toarray()
        categories['utterance'] = vectorizer.get_feature_names_out()
        logger.info(
            'Utterance have be transformed using TfidfVectorizer.Train %s.'
            'Val: %s. Test: %s',
            X_train.shape, X_val.shape, X_test.shape,
        )

    # Tokenization by Word
    elif utterance_processing == 'word':
        ...

    # BPE Tokenization
    elif utterance_processing == 'BPE':
        logger.info('Utterance will be transformed using Byte Pair Encoding')
        bpe_tokenizer = TokenizerBPE(
            vocab_size=50000,
            min_frequency=0,
            special_tokens=["[UNK]", "[PAD]"],
            continuing_subword_prefix="_",
            end_of_word_suffix="__",
            max_token_length=None,
            show_progress=True,
            unk_token="[UNK]",
            path='vocab.json',
        )

        # Prepare BPE tokens
        bpe_tokenizer.fit(df_train['Utterance'])
        X_train = bpe_tokenizer.transform(
            df_train['Utterance'],
            padding=True,
        )
        X_val = bpe_tokenizer.transform(
            df_val['Utterance'],
            padding=True,
            padding_size=X_train.shape[1],
        )
        X_test = bpe_tokenizer.transform(
            df_test['Utterance'],
            padding=True,
            padding_size=X_train.shape[1],
        )
        logger.info(
            'Utterance have be transformed using Byte Pair Encoding. Train %s.'
            'Val: %s. Test: %s',
            X_train.shape, X_val.shape, X_test.shape,
        )

    #######################################################################
    # Convert Speaker columns to One-Hot vectors
    if encode_speakers:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df_train.loc[:, 'Speaker'].values.reshape(-1, 1))
        categories['speakers'] = encoder.categories_
        logger.info('Speakers categories: %s', encoder.categories_)

        speaker_train = encoder.transform(
            df_train.loc[:, 'Speaker'].values.reshape(-1, 1)
        )
        speaker_val = encoder.transform(
            df_val.loc[:, 'Speaker'].values.reshape(-1, 1)
        )
        speaker_test = encoder.transform(
            df_test.loc[:, 'Speaker'].values.reshape(-1, 1)
        )
        # df_train.drop(columns=['Speaker'], inplace=True)
        # df_val.drop(columns=['Speaker'], inplace=True)
        # df_test.drop(columns=['Speaker'], inplace=True)

        logger.info(
            'Speaker One-Hot vectors have been prepared. Train: %s. Val: %s. Test: %s',
            speaker_train.shape, speaker_val.shape, speaker_test.shape,
        )

        # Concatenate to the dataset
        X_train = np.concatenate([X_train, speaker_train], axis=1)
        X_val = np.concatenate([X_val, speaker_val], axis=1)
        X_test = np.concatenate([X_test, speaker_test], axis=1)

        logger.info(
            'Speaker One-Hot vectors have been added to datasets. Train: %s.'
            'Val: %s. Test: %s',
            X_train.shape, X_val.shape, X_test.shape,
        )

    #######################################################################
    # Extract labels
    y_train = df_train.loc[:, labels].values
    y_val = df_val.loc[:, labels].values
    y_test = df_test.loc[:, labels].values
    logger.info(
        'Labels %s have been prepared. Train: %s. Val: %s. Test: %s',
        labels, y_train.shape, y_val.shape, y_test.shape,
    )

    ######################################################################
    # Convert NumPy arrays to PyTorch tensors
    logger.info('NumPy arrays is being converted to PyTorch tensors')
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    logger.info(
        'NumPy arrays have been converted to PyTorch tensors. X_train: %s.'
        'y_train: %s. X_val: %s. y_val: %s. X_test: %s. y_test: %s',
        X_train.size(), y_train.size(),
        X_val.size(), y_val.size(),
        X_test.size(), y_test.size(),
    )

    # Place Tensors to Dataset
    logger.info('Tensors is being placed to DataLoaders')
    ds_train = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    ds_val = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    ds_test = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    logger.info('Tensors have been placed to DataLoaders')

    return ds_train, ds_val, ds_test, categories
