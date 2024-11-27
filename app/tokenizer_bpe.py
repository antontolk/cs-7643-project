import logging
import json
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class TokenizerBPE:
    def __init__(
            self,
            vocab_size: int,
            min_frequency: int,
            special_tokens: list[str],
            continuing_subword_prefix: str,
            end_of_word_suffix: str,
            max_token_length: int | None,
            show_progress: bool,
            unk_token: str,
            path: str | None
    ):
        """Tokenizer based on Byte Pair Encoding.

        :param vocab_size: The size of the final vocabulary, including all
          tokens and alphabet.
        :type vocab_size: int
        :param min_frequency: The minimum frequency a pair should have in order
          to be merged.
        :type min_frequency: int
        :param special_tokens: A list of special tokens the model should know
          of.
        :type special_tokens: list[str]
        :param continuing_subword_prefix: A prefix to be used for every subword
          that is not a beginning-of-word.
        :type continuing_subword_prefix: str
        :param end_of_word_suffix: A suffix to be used for every subword that
          is a end-of-word.
        :type end_of_word_suffix: str
        :param max_token_length: Prevents creating tokens longer than the
          specified size. This can help with reducing polluting your vocabulary
          with highly repetitive tokens like ====== for wikipedia
        :type max_token_length: int | None
        :param show_progress: Whether to show progress bars while training.
        :type show_progress: bool
        :param unk_token: The token used for out-of-vocabulary tokens.
        :type unk_token: str
        :param path: The path where tokenizer config will be saved.
        :type path: str | None
        """
        self.show_progress = show_progress
        self.path = path
        self.voc_size = None

        # Init tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))

        # Add pretokenizer
        self.tokenizer.pre_tokenizer = Whitespace()

        # Initiate a trainer
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            max_token_length=max_token_length,
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

        # Train tokenizer
        self.tokenizer.train_from_iterator(
            iterator=X,
            trainer=self.trainer,
        )

        # Save vocabulary size
        if self.show_progress:
            self.voc_size = len(self.tokenizer.get_vocab())
            print(f'The vocabulary size: {self.voc_size}')

        if self.path is not None:
            self.tokenizer.save(self.path)

    def transform(
            self,
            X: pd.DataFrame | pd.Series,
            padding: bool,
            padding_size: None = None | int
    ) -> list[np.ndarray] | np.ndarray:
        """Tokenizes the text

        :param X: Dataset.
        :type X: pandas.DataFrame | pandas.Series
        :param padding: if True, all arrays will be padded
        :type padding: bool
        :param padding_size: If specified, the size of padding
        :type padding_size: int

        :return: tokenized dataset
        :rtype: list[np.ndarray] | np.ndarray
        """
        # Tokenize dataset
        out = [
            np.array(enc.ids)
            for enc in self.tokenizer.encode_batch(input=X)
        ]

        # Pad arrays by "[PAD]"
        if padding:
            pad_token = self.tokenizer.encode('[PAD]').ids[0]

            if isinstance(padding_size, int):
                max_length = padding_size
            else:
                max_length = max(x.shape[0] for x in out)

            out = np.array([
                np.pad(x, (0, max_length - x.shape[0]), constant_values=pad_token)
                for x in out
            ])

        return out
