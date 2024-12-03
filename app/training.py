"""The model training functions"""
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)


def model_training(
        model: nn.Module,
        dl_train: DataLoader,
        dl_val: DataLoader,
        dl_test: DataLoader,
        epochs: int,
        criterion_type: str,
        lr: float,
):
    """
    The model training.

    :param model: the model to be trained.
    :param dl_train: Train DataLoader
    :param dl_val: Validation DataLoader
    :param dl_test: Test DataLoader
    :param epochs: the number of epochs.
    :param criterion_type: the Criterion type to be used:
        ce - Cross Entropy
    :param lr: learning rate

    :return:
    """

    device = 'mps' if torch.mps.is_available() else \
        'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Device Type: {device}")

    # Define the loss functions
    if criterion_type == 'ce':
        criterion_sentiment = nn.CrossEntropyLoss()
        criterion_emotion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Not supported criterion type.')

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, train_emotion_correct, train_sentiment_correct, train_total = 0.0, 0, 0, 0
        val_loss, val_emotion_correct, val_sentiment_correct, val_total = 0.0, 0, 0, 0

        # Training loop
        for batch_X, batch_y in dl_train:
            # Move batches to the device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            out_emotion, out_sentiment = model(batch_X)

            # Loss
            loss = criterion_emotion(out_emotion, batch_y[:, 0]) \
                + criterion_sentiment(out_sentiment, batch_y[:, 1])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Accuracy Emotions
            _, y_pred_emotion = torch.max(out_emotion, 1)
            train_emotion_correct += (y_pred_emotion == batch_y[:, 0]).sum().item()
            train_total += batch_y[:, 0].size(0)

            # Accuracy Sentiment
            _, y_pred_sentiment = torch.max(out_sentiment, 1)
            train_sentiment_correct += (y_pred_sentiment == batch_y[:, 1]).sum().item()


        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in dl_val:
                # Move batches to the device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                out_emotion, out_sentiment = model(batch_X)

                # Loss
                loss = criterion_emotion(out_emotion, batch_y[:, 0]) \
                       + criterion_sentiment(out_sentiment, batch_y[:, 1])

                # Accumulate loss
                val_loss += loss.item()

                # Accuracy Emotions
                _, y_pred_emotion = torch.max(out_emotion, 1)
                val_emotion_correct += (y_pred_emotion == batch_y[:, 0]).sum().item()
                val_total += batch_y[:, 0].size(0)

                # Accuracy Sentiment
                _, y_pred_sentiment = torch.max(out_sentiment, 1)
                val_sentiment_correct += (y_pred_sentiment == batch_y[:, 1]).sum().item()


        # Epoch loss and accuracy
        epoch_train_loss = train_loss / len(dl_train)
        epoch_train_emotion_accuracy = train_emotion_correct / train_total
        epoch_train_sentiment_accuracy = train_sentiment_correct / train_total
        epoch_val_loss = val_loss / len(dl_val)
        epoch_val_emotion_accuracy = val_emotion_correct / val_total
        epoch_val_sentiment_accuracy = val_sentiment_correct / val_total

        logger.info(
            'Epoch %d/%d | '
            'Training Loss: %.4f | '
            'Training Emotion Accuracy: %.4f | '
            'Training Sentiment Accuracy: %.4f | '
            'Validation Loss: %.4f | '
            'Validation Emotion Accuracy: %.4f | '
            'Validation Sentiment Accuracy: %.4f ',
            epoch + 1, epochs,
            epoch_train_loss,
            epoch_train_emotion_accuracy,
            epoch_train_sentiment_accuracy,
            epoch_val_loss,
            epoch_val_emotion_accuracy,
            epoch_val_sentiment_accuracy,
        )