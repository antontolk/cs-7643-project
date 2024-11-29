from pathlib import Path

from app.data_load import load_dataset

from app.dataset_preprocessing import meld_processing
from app.training import model_training
from app.model_fc import FullyConnectedNet
from app.tokenizer_bpe import TokenizerBPE
from app.tokenizer_word import TokenizerWord

from app.settings import Settings

if __name__ == '__main__':
    # Load settings
    settings = Settings()

    # Load dataset
    emotions, sentiments, df_train_x, df_train_y, df_val_x, df_val_y, \
        df_test_x, df_test_y = load_dataset(
            train_path=settings.data_load.meld_train,
            val_path=settings.data_load.meld_val,
            test_path=settings.data_load.meld_test,
            dataset='MELD',
        )

    # BPE Tokenization
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

    # Word Tokenization
    word_tokenizer = TokenizerWord(
        vocab_size=50000,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=True,
        unk_token="[UNK]",
        path='vocab_word.json',
    )

    # Prepare Word tokens
    word_tokenizer.fit(df_train_x['Utterance'])
    train_tokens_word = word_tokenizer.transform(
        df_train_x['Utterance'],
        padding=True
    )
    val_tokens_word = word_tokenizer.transform(
        df_val_x['Utterance'],
        padding=True
    )
    test_tokens_word = word_tokenizer.transform(
        df_test_x['Utterance'],
        padding=True)

    # Prepare BPE tokens
    bpe_tokenizer.fit(df_train_x['Utterance'])
    train_tokens = bpe_tokenizer.transform(
        df_train_x['Utterance'],
        padding=True,
    )
    val_tokens = bpe_tokenizer.transform(
        df_val_x['Utterance'],
        padding=True,
    )
    test_tokens = bpe_tokenizer.transform(
        df_test_x['Utterance'],
        padding=True,
    )

    




