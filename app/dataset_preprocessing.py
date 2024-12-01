"""Data processing realted scripts"""
import logging
import regex as re

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import spacy

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
        lemmatization: bool,
        remove_punc_signs: bool,
        strip: bool,
        utterance_processing: str,
        ngram: tuple,
        stop_words: list | str = 'english',
        top_n_speakers: int | None = None,
        tokens_in_sentence: int | None = None,
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
    :param lemmatization: if True, lemmatization will be applied.
    :type lemmatization: bool
    :param remove_punc_signs: if True, punctuations and signs will be removed.
    :type remove_punc_signs: bool
    :param utterance_processing: Options:
        ngram - if ngram is (1, 1) Bag of Word, else N-Grams
        BPE - Byte Pair Encoding
    :type utterance_processing: str
    :param ngram: The lower and upper boundary of the range of n-values for
        different word n-grams or char n-grams to be extracted.
    :type ngram: tuple
    :param stop_words: The list of stop words.
    :type: set | list
    :param tokens_in_sentence: If specified, all sentences will be padded or
        shrunk to the specified length. For Word and BPE tokenization only
    :param top_n_speakers: if specified, only top N speakers will be considered
        for the speaker One-Hot encoding
    :type top_n_speakers: int | None
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

    ###########################################################################
    # Remove punctuations and signs
    if remove_punc_signs:
        logger.info('Punctuations and signs are being removed.')
        regex = r'[\p{P}\p{S}]+'
        df_train['Utterance'] = df_train.loc[:, 'Utterance'].map(
            lambda x: re.sub(regex, '', x)
        )
        df_val['Utterance'] = df_val.loc[:, 'Utterance'].map(
            lambda x: re.sub(regex, '', x)
        )
        df_test['Utterance'] = df_test.loc[:, 'Utterance'].map(
            lambda x: re.sub(regex, '', x)
        )
        logger.info('Punctuations and signs have been removed.')

    ###########################################################################
    if strip:
        logger.info('Utterances are being stripped.')
        df_train['Utterance'] = df_train.loc[:, 'Utterance'].map(
            lambda x: x.strip()
        )
        df_val['Utterance'] = df_val.loc[:, 'Utterance'].map(
            lambda x: x.strip()
        )
        df_test['Utterance'] = df_test.loc[:, 'Utterance'].map(
            lambda x: x.strip()
        )
        logger.info('Utterances have been stripped.')

    #######################################################################
    # Lemmatization
    if lemmatization:
        logger.info('Utterances are being lemmatized')

        # Load English model (small)
        spacy_model = spacy.load('en_core_web_sm')

        df_train['Utterance'] = df_train.loc[:, 'Utterance'].apply(
            lambda x: ' '.join([token.lemma_ for token in spacy_model(x)])
        )
        df_val['Utterance'] = df_val.loc[:, 'Utterance'].apply(
            lambda x: ' '.join([token.lemma_ for token in spacy_model(x)])
        )
        df_test['Utterance'] = df_test.loc[:, 'Utterance'].apply(
            lambda x: ' '.join([token.lemma_ for token in spacy_model(x)])
        )
        logger.info('Utterances have been lemmatized')


    #######################################################################
    # Transformations of the Utterance column
    if utterance_processing == 'counts':
        logger.info(
            'Utterances are being transformed using CountVectorizer, N-Grams: %s',
            ngram,
        )
        count_vect = CountVectorizer(
            lowercase=True,
            ngram_range=ngram,
            stop_words=stop_words,
        )
        count_vect.fit(df_train.loc[:, 'Utterance'])
        X_train = count_vect.transform(df_train.loc[:, 'Utterance']).toarray()
        X_val = count_vect.transform(df_val.loc[:, 'Utterance']).toarray()
        X_test = count_vect.transform(df_test.loc[:, 'Utterance']).toarray()
        categories['utterance'] = count_vect.get_feature_names_out()
        logger.info(
            'Utterance have be transformed using CountVectorizer. '
            'Vocabulary size: %d. Train %s. Val: %s. Test: %s',
            len(count_vect.vocabulary_),
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
            stop_words=stop_words,
        )
        vectorizer.fit(df_train.loc[:, 'Utterance'])
        X_train = vectorizer.transform(df_train.loc[:, 'Utterance']).toarray()
        X_val = vectorizer.transform(df_val.loc[:, 'Utterance']).toarray()
        X_test = vectorizer.transform(df_test.loc[:, 'Utterance']).toarray()
        categories['utterance'] = vectorizer.get_feature_names_out()
        logger.info(
            'Utterance have be transformed using TfidfVectorizer.'
            'Vocabulary size: %d. Train %s.Val: %s. Test: %s',
            len(vectorizer.vocabulary_),
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
            tokens_in_sentence=tokens_in_sentence,
        )
        X_val = bpe_tokenizer.transform(
            df_val['Utterance'],
            tokens_in_sentence=tokens_in_sentence,
        )
        X_test = bpe_tokenizer.transform(
            df_test['Utterance'],
            tokens_in_sentence=tokens_in_sentence,
        )
        logger.info(
            'Utterance have be transformed using Byte Pair Encoding. Train %s.'
            'Val: %s. Test: %s',
            X_train.shape, X_val.shape, X_test.shape,
        )

    #######################################################################
    # Convert Speaker columns to One-Hot vectors
    if encode_speakers:

        # Leave only top N speakers, others will replaced by Other
        if top_n_speakers:
            top_n_speakers = df_train.loc[:, 'Speaker'].value_counts()\
                .nlargest(top_n_speakers).index
            df_train['Speaker'] = df_train.loc[:, 'Speaker'].where(
                df_train.loc[:, 'Speaker'].isin(top_n_speakers),
                other='Other',
            )
            df_val['Speaker'] = df_val.loc[:, 'Speaker'].where(
                df_val.loc[:, 'Speaker'].isin(top_n_speakers),
                other='Other',
            )
            df_test['Speaker'] = df_test.loc[:, 'Speaker'].where(
                df_test.loc[:, 'Speaker'].isin(top_n_speakers),
                other='Other',
            )

        encoder = OneHotEncoder(
            # max_categories=top_n_speakers,
            handle_unknown='ignore',
            sparse_output=False,
        )
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
    dl_train = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    dl_val = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    dl_test = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    logger.info('Tensors have been placed to DataLoaders')

    return dl_train, dl_val, dl_test, categories