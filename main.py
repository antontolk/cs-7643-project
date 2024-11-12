from app.data_load import load_dataset
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

    print(emotions)
    print(sentiments)
    print(df_train_x.head(5))
    print(df_train_x.dtypes)
    print(df_train_y.head(5))
    print(df_train_y.dtypes)

