from app.data_load import load_dataset
from app.settings import Settings
from app.tokenizer import tokenizer_by_word

if __name__ == '__main__':
    # Load settings
    settings = Settings()


    # Load dataset
    df_train, _, _ = load_dataset(
        train_path=settings.data_load.meld_train,
        val_path=settings.data_load.meld_val,
        test_path=settings.data_load.meld_test,
        dataset='MELD',
    )

    print(df_train.head(5))
    print(df_train.dtypes)
    
    df_train['Tokens'] = df_train['Utterance'].apply(tokenizer_by_word)

    print(df_train[['Utterance', 'Tokens']].head(5))