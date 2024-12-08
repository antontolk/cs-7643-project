from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from app import module_root


class DataLoadSettings(BaseSettings):
    """Data Load settings."""
    dataset_folder: str = Field()
    dataset: str = Field()
    train_filename: str = Field()
    val_filename: str = Field()
    test_filename: str = Field()

    @property
    def train(self) -> Path:
        """Returns a path to the MELD train dataset"""
        return module_root / '..' / self.dataset_folder / self.dataset / \
            self.train_filename

    @property
    def val(self) -> Path:
        """Returns a path to the MELD val dataset"""
        return module_root / '..' / self.dataset_folder / self.dataset / \
            self.val_filename

    @property
    def test(self) -> Path:
        """Returns a path to the MELD test dataset"""
        return module_root / '..' / self.dataset_folder / self.dataset / \
            self.test_filename


class DatasetProcessing(BaseSettings):
    """Dataset Processing Settings"""
    labels: list = Field()                  # Options: 'Emotion', 'Sentiment'
    utterance_processing: str = Field()     # Options: counts, TF-IDF, word, BPE
    lemmatization: bool = Field()
    ngram: tuple = Field()
    stop_words: str = Field()
    remove_punc_signs: bool = Field()       # Remove punctuation, signs
    strip: bool = Field()
    tokens_in_sentence: int = Field()       # The size of the sentence (BPE, Word only)
    encode_speakers: bool = Field()         # Will add speakers to samples
    top_n_speakers: int = Field()           # Only Top N speakers will be considered
    batch_size: int = Field()
    shuffle: bool = Field()


class ModelSettings(BaseSettings):
    """The Deep Learning Configuration"""
    # Model type. Options:
    # fc - Fully Connected
    # cnn - CNN1D
    type: str = Field()
    hidden_size: int = Field()      # The size of the hidden layer


class TrainingSettings(BaseSettings):
    """The Training Settings"""
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 1e-2

    # Criterion type. Options:
    # ce - Cross Entropy, 'wce' - Weighted Cross Entropy,
    # 'focal' - Focal Loss, 'label_smoothing' - Label Smoothing Loss
    criterion_type: str = 'wce'

    # Optimizer type. Options:
    # AdamW
    optimiser: str = Field()


class Settings(BaseSettings):
    """Application settings."""
    dev: bool = False
    config_file: str = 'config.json'
    output_dir: str = Field()

    data_load: DataLoadSettings = Field()
    data_preprocessing: DatasetProcessing = Field()
    model: ModelSettings = Field()
    training: TrainingSettings = Field()


    @property
    def output_dir_path(self) -> Path:
        """Returns a path to the output folder"""
        return module_root / '..' / self.output_dir

    @classmethod
    def load(
        cls,
        config_path: Path = module_root / '..' / 'config' / 'config.json',
    ):
        """Load the application configuration file."""
        return cls.parse_file(config_path)
