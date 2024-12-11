import logging

from app.data_load import load_dataset
from app.dataset_preprocessing import meld_processing
from app.bert_model import BertlNet
from app.model_fc import FullyConnectedNet
from app.model_cnn import CNN1DNet
from app.model_transformer import TransformerNet
from app.training import model_training
from app.settings import Settings
from app.visualisation import visualisation
from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)


if __name__ == '__main__':
    # Load settings
    settings = Settings.load()

    # Load dataset
    df_train, df_val, df_test = load_dataset(
            train_path=settings.data_load.train,
            val_path=settings.data_load.val,
            test_path=settings.data_load.test,
            dataset=settings.data_load.dataset,
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
        batch_size=settings.data_preprocessing.batch_size,
        bert_model_name=settings.data_preprocessing.bert_model_name,
    )

    # Create the model
    if settings.model.type == 'fc':
        model = FullyConnectedNet(
            n_features=dl_train.dataset[0][0].shape[0] + dl_train.dataset[0][1].shape[0],
            labels=settings.data_preprocessing.labels,
            hidden_size=settings.model.hidden_size,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ],
            dropout_rate=settings.model.dropout_rate,
            use_batch_norm=settings.model.batch_norm,
        )
     
    elif settings.model.type == 'cnn':
        model = CNN1DNet(
            vocab_size=categories['vocab_size'],
            embedding_dim=settings.model.embedding_dim,
            kernel_sizes=settings.model.kernel_sizes,
            num_filters=settings.model.num_filters,
            dropout_rate=settings.model.dropout_rate,
            labels=settings.data_preprocessing.labels,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ],
            n_speakers=settings.data_preprocessing.top_n_speakers + 1,
            hidden_size=settings.model.hidden_size,
            use_batch_norm=settings.model.batch_norm,
        )
        logger.info('CNN initiated. \n %s', model)
    elif settings.model.type == 'transformer':
        model = TransformerNet(
            vocab_size=categories['vocab_size'],
            embedding_dim=settings.model.embedding_dim,
            n_speakers=settings.data_preprocessing.top_n_speakers + 1,
            n_classes=[
                len(categories['emotions']),
                len(categories['sentiments']),
            ],
            n_heads=settings.model.n_heads,
            n_layers=settings.model.n_layers,
            hidden_size=settings.model.hidden_size,
            dropout_rate=settings.model.dropout_rate,
            labels=settings.data_preprocessing.labels,
            # max_len=max(data[0].shape[0] for data in dl_train.dataset),
        )
        logger.info('Fully Connected model initiated. \n %s', model)
    elif settings.model.type == 'bert':
        model = BertlNet(
            bert_model_name=settings.data_preprocessing.bert_model_name,
            n_speakers=settings.data_preprocessing.top_n_speakers + 1,
            n_emotion_classes=len(categories['emotions']),
            n_sentiment_classes=len(categories['sentiments']),
            dropout_rate=settings.model.dropout_rate,
        )
    else:
        raise ValueError('Not supported model type.')

    df_results, cm, best_model = model_training(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        epochs=settings.training.epochs,
        criterion_type=settings.training.criterion_type,
        lr=settings.training.lr,
        weight_decay=settings.training.weight_decay,
        labels=settings.data_preprocessing.labels,
        n_classes=[
            len(categories['emotions']),
            len(categories['sentiments']),
        ],
        utterance_processing=settings.data_preprocessing.utterance_processing,
        seed=123,
    )
    
    visualisation(
        df=df_results,
        cm=cm,
        labels=settings.data_preprocessing.labels,
        output_dir=settings.output_dir_path,
    )


    # torch.save(model, "cnn1d_model.pth") # trained on CUDA gpu device

    # this is to load on cpu
    #device = torch.device("cpu")
    #model.load_state_dict(torch.load("cnn1d_model.pth", map_location=device))
