from pathlib import Path

from app.data_load import load_dataset
from app.dataset_preprocessing import meld_processing


from app.settings import Settings

if __name__ == '__main__':
    # Load settings
    settings = Settings()

    # Load dataset
    df_train, df_val, df_test = load_dataset(
            train_path=settings.data_load.meld_train,
            val_path=settings.data_load.meld_val,
            test_path=settings.data_load.meld_test,
            dataset='MELD',
        )

    # Data preprocessing
    ds_train, ds_val, ds_test, categories = meld_processing(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        labels=settings.data_preprocessing.labels,
        encode_speakers=settings.data_preprocessing.encode_speakers,
        utterance_processing=settings.data_preprocessing.utterance_processing,
        ngram=settings.data_preprocessing.ngram,
    )

    # Create the model

    # Train the model



