import logging

from app.data_load import load_dataset
from app.bert_training import bert_model_training
from app.bert_model import BERT_Arch
from transformers import AutoModel, BertTokenizerFast

from app.tokenizer_word import TokenizerWord


from app.dataset_preprocessing import meld_processing
from app.training import model_training
from app.model_fc import FullyConnectedNet
from app.settings import Settings
from app.visualisation import visualisation
from app.logging_config import logger_config
import argparse

logger = logging.getLogger(__name__)
logger_config(logger)

class BertModelTrainer:
    @staticmethod
    def train_bert_model(settings, dl_train, dl_val, dl_test, categories):
        bert = AutoModel.from_pretrained('bert-base-uncased')
        model = BERT_Arch(
            bert,
            labels=settings.data_preprocessing.labels,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ]
        )
        bert_model_training(
            model=model,
            dl_train=dl_train,
            dl_val=dl_val,
            dl_test=dl_test,
            epochs=settings.bert_training.epochs,
            criterion_type=settings.bert_training.criterion_type,
            lr=settings.bert_training.lr,
            optimiser_val=settings.bert_training.optimiser_val
        )

        # visualize here


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    parser.add_argument('-t','--type', help='Choose the model, fc and bert are only supported options', required=False)
    args = vars(parser.parse_args())
    # Load settings
    settings = Settings()
    if args['type'] is not None:
        settings.model.type = args['type']

    print('Model type:', settings.model.type)

    # Load dataset
    df_train, df_val, df_test = load_dataset(
            train_path=settings.data_load.meld_train,
            val_path=settings.data_load.meld_val,
            test_path=settings.data_load.meld_test,
            dataset='MELD',
        )

    # Data preprocessing
    dl_train, dl_val, dl_test, categories = meld_processing(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        labels=settings.data_preprocessing.labels,
        utterance_processing=settings.data_preprocessing.utterance_processing,
        remove_punc_signs=settings.data_preprocessing.remove_punc_signs,
        strip=settings.data_preprocessing.strip,
        lemmatization=settings.data_preprocessing.lemmatization,
        ngram=settings.data_preprocessing.ngram,
        stop_words=settings.data_preprocessing.stop_words,
        tokens_in_sentence=settings.data_preprocessing.tokens_in_sentence,
        encode_speakers=settings.data_preprocessing.encode_speakers,
        top_n_speakers=settings.data_preprocessing.top_n_speakers,
    )

    # Word Tokenization
    word_tokenizer = TokenizerWord(
        vocab_size=50000,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=True,
        unk_token="[UNK]",
        path='vocab_word.json',
    )

    # Create the model

    if settings.model.type == 'fc':
        model = FullyConnectedNet(
            n_features=dl_train.dataset[0][0].shape[0],
            labels=settings.data_preprocessing.labels,
            hidden=settings.model.hidden_size,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ],
        )
        logger.info(model)
        # Train the model
        df_results, cm = model_training(
            model=model,
            dl_train=dl_train,
            dl_val=dl_val,
            dl_test=dl_test,
            epochs=settings.training.epochs,
            criterion_type=settings.training.criterion_type,
            lr=settings.training.lr,
            weight_decay=settings.training.weight_decay,
            labels=settings.training.labels,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ],
        )

        visualisation(
            df=df_results,
            cm=cm,
            labels=settings.training.labels,
            output_dir=settings.output_dir_path,
        )
    elif settings.model.type == 'bert':
        BertModelTrainer().train_bert_model(settings, dl_train, dl_val, dl_test, categories)
    else:
        raise ValueError('Not supported model type.')

