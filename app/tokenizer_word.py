import logging
import json
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

logger = logging.getLogger(__name__)

class TokenizerWord:
    def __init__(
            self,
            vocab_size: int,
            special_tokens: list[str],
            show_progress: bool,
            unk_token: str,
            path: str | None
    ):
        """Tokenizer based on Byte Pair Encoding.

        :param vocab_size: The size of the final vocabulary, including all
          tokens and alphabet.
        :type vocab_size: int
        :param special_tokens: A list of special tokens the model should know
          of.
        :type special_tokens: list[str]
        :param show_progress: Whether to show progress bars while training.
        :type show_progress: bool
        :param unk_token: The token used for out-of-vocabulary tokens.
        :type unk_token: str
        :param path: The path where tokenizer config will be saved.
        :type path: str | None
        """
        self.show_progress = show_progress
        self.path = path
        self.vocab_size = vocab_size

        # Init tokenizer
        self.tokenizer = Tokenizer(WordLevel(unk_token=unk_token))

        # Add pretokenizer
        self.tokenizer.pre_tokenizer = Whitespace()

        # Initiate a trainer
        self.trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )


    def fit(
            self,
            X: pd.DataFrame | pd.Series,
    ):
        """Fits the tokenizer.

        :param X: Dataset.
        :type X: pandas.DataFrame | pandas.Series

        :return: A tuple with train, validation, test Datasets and vocabulary sizes
          on Success, or None.
        :rtype: tuple[Dataset, Dataset, Dataset, dict[str, int]]
        """

        logger.info('Training Word-Level Tokenizer...')

        # Train tokenizer
        self.tokenizer.train_from_iterator(
            iterator=X,
            trainer=self.trainer,
        )

        logger.info('Word-Level Tokenizer training completed.')

        if self.path is not None:
            self.tokenizer.save(self.path)
            logger.info('Tokenizer saved at %s.', self.path)

    def transform(
            self,
            X: pd.DataFrame | pd.Series,
            tokens_in_sentence: int | None = None,
    ) -> list[np.ndarray] | np.ndarray:
        """Tokenizes the text

        :param X: Dataset.
        :type X: pandas.DataFrame | pandas.Series
        :param padding: if True, all arrays will be padded

        :return: tokenized dataset
        :rtype: list[np.ndarray] | np.ndarray
        """
        
        logger.info('Transforming text data using Word-Level Tokenizer...')

        # Tokenize dataset
        tokenized = [
            np.array(enc.ids)
            for enc in self.tokenizer.encode_batch(input=X.tolist())
        ]

        # Pad arrays or truncate
        if tokens_in_sentence:
            pad_token = self.tokenizer.token_to_id('[PAD]')
            padded = np.zeros((len(tokenized), tokens_in_sentence), dtype='int64')

            for i, tokens in enumerate(tokenized):
                if len(tokens) < tokens_in_sentence:
                    padded[i] = np.pad(
                        tokens,
                        (0, tokens_in_sentence - len(tokens)),
                        constant_values=pad_token,
                    )
                else:
                    padded[i] = tokens[:tokens_in_sentence]

            logger.info('Text data transformed and padded to %d tokens.', tokens_in_sentence)
            return padded

        logger.info('Text data transformed without padding.')
        return np.array(tokenized, dtype=object)
