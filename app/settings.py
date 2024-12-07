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
    utterance_processing: str = 'word'     # Options: counts, TF-IDF, word, BPE
    lemmatization: bool = True
    ngram: tuple = (1, 5)
    stop_words: str = 'english'
    remove_punc_signs: bool = False          # Remove punctuation, signs
    strip: bool = True
    tokens_in_sentence: int = 30            # The size of the sentence (BPE, Word only)
    encode_speakers: bool = True            # Will add speakers to samples
    top_n_speakers: int = 10                # Only Top N speakers will be considered
    batch_size: int = 32
    shuffle: bool = True


class ModelSettings(BaseSettings):
    """The Deep Learning Configuration"""
    type: str = 'transformer'        # transformer
    hidden_size: int = 16      # The size of the hidden layer


class TrainingSettings(BaseSettings):
    """The Training Settings"""
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 1e-2
    # ce - Cross Entropy, 'wce' - Weighted Cross Entropy,
    # 'focal' - Focal Loss, 'label_smoothing' - Label Smoothing Loss
    criterion_type: str = 'wce'
    labels: list = ['Emotion', 'Sentiment']  # Options: 'Emotion', 'Sentiment'


class Settings(BaseSettings):
    """Application settings."""
    dev: bool = False
    output_dir: str = 'results'

    data_load: DataLoadSettings = DataLoadSettings()
    data_preprocessing: DatasetProcessing = DatasetProcessing()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()

    @property
    def output_dir_path(self) -> Path:
        """Returns a path to the output folder"""
        return module_root / '..' / self.output_dir
