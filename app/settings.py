from pathlib import Path

from pydantic_settings import BaseSettings

from app import module_root

class DataLoadSettings(BaseSettings):
    """Data Load settings."""
    dataset_folder: str = 'dataset'
    MELD_dataset: str = 'MELD'
    MELD_train_filename: str = 'train_sent_emo.csv'
    MELD_val_filename: str = 'dev_sent_emo.csv'
    MELD_test_filename: str = 'test_sent_emo.csv'

    @property
    def meld_train(self) -> Path:
        """Returns a path to the MELD train dataset"""
        return module_root / '..' / self.dataset_folder / self.MELD_dataset / \
            self.MELD_train_filename

    @property
    def meld_val(self) -> Path:
        """Returns a path to the MELD val dataset"""
        return module_root / '..' / self.dataset_folder / self.MELD_dataset / \
            self.MELD_val_filename

    @property
    def meld_test(self) -> Path:
        """Returns a path to the MELD test dataset"""
        return module_root / '..' / self.dataset_folder / self.MELD_dataset / \
            self.MELD_test_filename


class DatasetProcessing(BaseSettings):
    """Dataset Processing Settings"""
    labels: list = ['Emotion', 'Sentiment']  # Options: 'Emotion', 'Sentiment'
    encode_speakers: bool = True            # Will add speakers to samples
    utterance_processing: str = 'TF-IDF'     # Options: word, BPE, counts, TF-IDF
    ngram: tuple = (1, 2)
    batch_size: int = 32
    shuffle: bool = True


class ModelSettings(BaseSettings):
    """The Deep Learning Configuration"""
    ...


class TrainingSettings(BaseSettings):
    """The Training Settings"""
    epochs: int = 20


class Settings(BaseSettings):
    """Application settings."""
    dev: bool = False

    data_load: DataLoadSettings = DataLoadSettings()
    data_preprocessing: DatasetProcessing = DatasetProcessing()