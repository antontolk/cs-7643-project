"""The model training functions"""
import logging

import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix

import matplotlib.pyplot as plt
import seaborn as sns

from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)


class FocalLoss(nn.Module):
    """Focal loss"""
    def __init__(
            self,
            alpha,
            gamma,
            weight=None,
            reduction='mean',
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(
            weight=self.weight,
            reduction='none',
        )(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, logits, target):
        pred = logits.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def model_training(
        model: nn.Module,
        dl_train: DataLoader,
        dl_val: DataLoader,
        dl_test: DataLoader,
        epochs: int,
        criterion_type: str,
        lr: float,
        weight_decay: float,
        labels: list,
        n_classes: list,
        alpha: float = 1.0,
        gamma: float = 2.0,
        smoothing: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """
    The model training.

    :param model: the model to be trained.
    :param dl_train: Train DataLoader
    :param dl_val: Validation DataLoader
    :param dl_test: Test DataLoader
    :param epochs: the number of epochs.
    :param criterion_type: the Criterion type to be used:
        'ce' - Cross Entropy
        'wce' - Weighted Cross Entropy
        'focal' - Focal Loss
        'label_smoothing' - Label Smoothing Loss
    :param lr: learning rate
    :param weight_decay: L2 regularization
    :param labels: List of label names (e.g., ['Emotion', 'Sentiment'])
    :param n_classes: List of number of classes for each label
    :param alpha: alpha parameter for Focal Loss (default: 1.0)
    :param gamma: gamma parameter for Focal Loss (default: 2.0)
    :param smoothing: smoothing parameter for Label Smoothing Loss (default: 0.1)

    :return: training results and confusion matrices
    """

    device = 'mps' if torch.mps.is_available() else \
        'cuda' if torch.cuda.is_available() else 'cpu'

    n_emotion_classes = n_classes[labels.index('Emotion')]
    n_sentiment_classes = n_classes[labels.index('Sentiment')]

    # Collect the training labels to compute class weights
    emotion_labels = []
    sentiment_labels = []
    for _, batch_y in dl_train:
        emotion_labels.append(batch_y[:, 0])
        sentiment_labels.append(batch_y[:, 1])
    emotion_labels = torch.cat(emotion_labels)
    sentiment_labels = torch.cat(sentiment_labels)

    # Define the loss functions
    if criterion_type in ['wce', 'focal']:
        # Compute class weights for emotions
        emotion_class_counts = torch.bincount(emotion_labels)
        emotion_class_weights = 1.0 / emotion_class_counts.float()
        emotion_class_weights = emotion_class_weights / emotion_class_weights.sum()
        emotion_class_weights = emotion_class_weights.to(device)

        # Compute class weights for sentiments
        sentiment_class_counts = torch.bincount(sentiment_labels)
        sentiment_class_weights = 1.0 / sentiment_class_counts.float()
        sentiment_class_weights = sentiment_class_weights / sentiment_class_weights.sum()
        sentiment_class_weights = sentiment_class_weights.to(device)

    if criterion_type == 'ce':
        criterion_sentiment = nn.CrossEntropyLoss()
        criterion_emotion = nn.CrossEntropyLoss()
    elif criterion_type == 'wce':
        criterion_emotion = nn.CrossEntropyLoss(weight=emotion_class_weights)
        criterion_sentiment = nn.CrossEntropyLoss(weight=sentiment_class_weights)
    elif criterion_type == 'focal':
        criterion_emotion = FocalLoss(alpha=alpha, gamma=gamma, weight=emotion_class_weights)
        criterion_sentiment = FocalLoss(alpha=alpha, gamma=gamma, weight=sentiment_class_weights)
    elif criterion_type == 'label_smoothing':
        criterion_emotion = LabelSmoothingLoss(classes=n_emotion_classes, smoothing=0.1)
        criterion_sentiment = LabelSmoothingLoss(classes=n_sentiment_classes, smoothing=0.1)
    else:
        raise ValueError('Not supported criterion type.')

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.to(device)

    # Initialize DataFrame to store training results
    df = pd.DataFrame()

    for epoch in range(epochs):
        #######################################################################
        # Training loop
        model.train()
        train_loss = 0.0
        train_emotion_correct = 0
        train_sentiment_correct = 0
        train_total = 0

        # Initialize metrics
        train_emotion_f1_macro = MulticlassF1Score(
            num_classes=n_emotion_classes,
            average='macro',
        ).to(device)
        train_sentiment_f1_macro = MulticlassF1Score(
            num_classes=n_sentiment_classes,
            average='macro',
        ).to(device)
        train_emotion_f1_weighted = MulticlassF1Score(
            num_classes=n_emotion_classes,
            average='weighted',
        ).to(device)
        train_sentiment_f1_weighted = MulticlassF1Score(
            num_classes=n_sentiment_classes,
            average='weighted',
        ).to(device)

        for batch_X, batch_y in dl_train:
            # Move batches to the device
            batch_X = batch_X.to(device)
            emotion_labels = batch_y[:, 0].to(device)
            sentiment_labels = batch_y[:, 1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            out_emotion, out_sentiment = model(batch_X)

            # Loss
            loss = criterion_emotion(out_emotion, emotion_labels) \
                   + criterion_sentiment(out_sentiment, sentiment_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Predictions
            _, y_pred_emotion = torch.max(out_emotion, 1)
            _, y_pred_sentiment = torch.max(out_sentiment, 1)

            # Correct counts
            train_emotion_correct += (
                    y_pred_emotion == emotion_labels
            ).sum().item()
            train_sentiment_correct += (
                    y_pred_sentiment == sentiment_labels
            ).sum().item()
            train_total += batch_y.size(0)

            # Update metrics
            train_emotion_f1_macro.update(y_pred_emotion, emotion_labels)
            train_sentiment_f1_macro.update(y_pred_sentiment, sentiment_labels)
            train_emotion_f1_weighted.update(y_pred_emotion, emotion_labels)
            train_sentiment_f1_weighted.update(y_pred_sentiment, sentiment_labels)

        # Compute training metrics
        epoch_train_loss = train_loss / len(dl_train)
        epoch_train_emotion_accuracy = train_emotion_correct / train_total
        epoch_train_sentiment_accuracy = train_sentiment_correct / train_total

        train_emotion_macro_f1 = train_emotion_f1_macro.compute().item()
        train_sentiment_macro_f1 = train_sentiment_f1_macro.compute().item()
        train_emotion_weighted_f1 = train_emotion_f1_weighted.compute().item()
        train_sentiment_weighted_f1 = train_sentiment_f1_weighted.compute().item()

        # Log the metrics
        logger.info(
            'Training: Epoch %d/%d | '
            'Loss: %.4f | '
            'Emotion Accuracy: %.4f | '
            'Sentiment Accuracy: %.4f | '
            'Emotion Macro F1: %.4f | '
            'Sentiment Macro F1: %.4f | '
            'Emotion Weighted F1: %.4f | '
            'Sentiment Weighted F1: %.4f | ',
            epoch + 1, epochs,
            epoch_train_loss,
            epoch_train_emotion_accuracy,
            epoch_train_sentiment_accuracy,
            train_emotion_macro_f1,
            train_sentiment_macro_f1,
            train_emotion_weighted_f1,
            train_sentiment_weighted_f1,
        )

        # Save metrics to DataFrame
        df_train = pd.DataFrame({
            'epoch': [epoch + 1],
            'type': ['train'],
            'loss': [epoch_train_loss],
            'emotion_accuracy': [epoch_train_emotion_accuracy],
            'sentiment_accuracy': [epoch_train_sentiment_accuracy],
            'emotion_macro_f1': [train_emotion_macro_f1],
            'sentiment_macro_f1': [train_sentiment_macro_f1],
            'emotion_weighted_f1': [train_emotion_weighted_f1],
            'sentiment_weighted_f1': [train_sentiment_weighted_f1],
        })

        # Reset metrics
        train_emotion_f1_macro.reset()
        train_sentiment_f1_macro.reset()
        train_emotion_f1_weighted.reset()
        train_sentiment_f1_weighted.reset()

        #######################################################################
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_emotion_correct = 0
        val_sentiment_correct = 0
        val_total = 0

        val_emotion_f1_macro = MulticlassF1Score(
            num_classes=n_emotion_classes,
            average='macro',
        ).to(device)
        val_sentiment_f1_macro = MulticlassF1Score(
            num_classes=n_sentiment_classes,
            average='macro',
        ).to(device)
        val_emotion_f1_weighted = MulticlassF1Score(
            num_classes=n_emotion_classes,
            average='weighted',
        ).to(device)
        val_sentiment_f1_weighted = MulticlassF1Score(
            num_classes=n_sentiment_classes,
            average='weighted',
        ).to(device)

        with torch.no_grad():
            for batch_X, batch_y in dl_val:
                # Move batches to the device
                batch_X = batch_X.to(device)
                emotion_labels = batch_y[:, 0].to(device)
                sentiment_labels = batch_y[:, 1].to(device)

                # Forward pass
                out_emotion, out_sentiment = model(batch_X)

                # Loss
                loss = criterion_emotion(out_emotion, emotion_labels) \
                       + criterion_sentiment(out_sentiment, sentiment_labels)

                # Accumulate loss
                val_loss += loss.item()

                # Predictions
                _, y_pred_emotion = torch.max(out_emotion, 1)
                _, y_pred_sentiment = torch.max(out_sentiment, 1)

                # Correct counts
                val_emotion_correct += (
                        y_pred_emotion == emotion_labels
                ).sum().item()
                val_sentiment_correct += (
                        y_pred_sentiment == sentiment_labels
                ).sum().item()
                val_total += batch_y.size(0)

                # Update metrics
                val_emotion_f1_macro.update(y_pred_emotion, emotion_labels)
                val_sentiment_f1_macro.update(y_pred_sentiment, sentiment_labels)
                val_emotion_f1_weighted.update(y_pred_emotion, emotion_labels)
                val_sentiment_f1_weighted.update(y_pred_sentiment, sentiment_labels)

        # Compute validation metrics
        epoch_val_loss = val_loss / len(dl_val)
        epoch_val_emotion_accuracy = val_emotion_correct / val_total
        epoch_val_sentiment_accuracy = val_sentiment_correct / val_total


        val_emotion_macro_f1 = val_emotion_f1_macro.compute().item()
        val_sentiment_macro_f1 = val_sentiment_f1_macro.compute().item()
        val_emotion_weighted_f1 = val_emotion_f1_weighted.compute().item()
        val_sentiment_weighted_f1 = val_sentiment_f1_weighted.compute().item()

        # Reset metrics
        val_emotion_f1_macro.reset()
        val_sentiment_f1_macro.reset()
        val_emotion_f1_weighted.reset()
        val_sentiment_f1_weighted.reset()

        # Log the metrics
        logger.info(
            'Validation: Epoch %d/%d | '
            'Loss: %.4f | '
            'Emotion Accuracy: %.4f | '
            'Sentiment Accuracy: %.4f | '
            'Emotion Macro F1: %.4f | '
            'Sentiment Macro F1: %.4f | '
            'Emotion Weighted F1: %.4f | '
            'Sentiment Weighted F1: %.4f',
            epoch + 1, epochs,
            epoch_val_loss,
            epoch_val_emotion_accuracy,
            epoch_val_sentiment_accuracy,
            val_emotion_macro_f1,
            val_sentiment_macro_f1,
            val_emotion_weighted_f1,
            val_sentiment_weighted_f1,
        )

        df_val = pd.DataFrame({
            'epoch': [epoch + 1],
            'type': ['val'],
            'loss': [epoch_val_loss],
            'emotion_accuracy': [epoch_val_emotion_accuracy],
            'sentiment_accuracy': [epoch_val_sentiment_accuracy],
            'emotion_macro_f1': [val_emotion_macro_f1],
            'sentiment_macro_f1': [val_sentiment_macro_f1],
            'emotion_weighted_f1': [val_emotion_weighted_f1],
            'sentiment_weighted_f1': [val_sentiment_weighted_f1],
        })
        df = pd.concat([df, df_train, df_val])

    ###########################################################################
    # Test loop
    model.eval()
    test_loss = 0.0
    test_emotion_correct = 0
    test_sentiment_correct = 0
    test_total = 0

    test_emotion_f1_macro = MulticlassF1Score(
        num_classes=n_emotion_classes,
        average='macro',
    ).to(device)
    test_sentiment_f1_macro = MulticlassF1Score(
        num_classes=n_sentiment_classes,
        average='macro',
    ).to(device)
    test_emotion_f1_weighted = MulticlassF1Score(
        num_classes=n_emotion_classes,
        average='weighted',
    ).to(device)
    test_sentiment_f1_weighted = MulticlassF1Score(
        num_classes=n_sentiment_classes,
        average='weighted',
    ).to(device)

    test_emotion_cm = MulticlassConfusionMatrix(
        num_classes=n_emotion_classes,
    ).to(device)
    test_sentiment_cm = MulticlassConfusionMatrix(
        num_classes=n_sentiment_classes,
    ).to(device)

    with torch.no_grad():
        for batch_X, batch_y in dl_test:
            batch_X = batch_X.to(device)
            emotion_labels = batch_y[:, 0].to(device)
            sentiment_labels = batch_y[:, 1].to(device)

            out_emotion, out_sentiment = model(batch_X)

            # Loss
            loss = criterion_emotion(out_emotion, emotion_labels) \
                   + criterion_sentiment(out_sentiment, sentiment_labels)

            # Accumulate loss
            test_loss += loss.item()

            # Predictions
            _, y_pred_emotion = torch.max(out_emotion, 1)
            _, y_pred_sentiment = torch.max(out_sentiment, 1)

            # Correct counts
            test_emotion_correct += (
                    y_pred_emotion == emotion_labels
            ).sum().item()
            test_sentiment_correct += (
                    y_pred_sentiment == sentiment_labels
            ).sum().item()
            test_total += batch_y.size(0)

            # Update metrics
            test_emotion_f1_macro.update(y_pred_emotion, emotion_labels)
            test_sentiment_f1_macro.update(y_pred_sentiment, sentiment_labels)
            test_emotion_f1_weighted.update(y_pred_emotion, emotion_labels)
            test_sentiment_f1_weighted.update(y_pred_sentiment, sentiment_labels)

            # Update confusion matrices
            test_emotion_cm.update(y_pred_emotion, emotion_labels)
            test_sentiment_cm.update(y_pred_sentiment, sentiment_labels)

    # Compute test metrics
    epoch_test_loss = test_loss / len(dl_test)
    epoch_test_emotion_accuracy = test_emotion_correct / test_total
    epoch_test_sentiment_accuracy = test_sentiment_correct / test_total

    test_emotion_macro_f1 = test_emotion_f1_macro.compute().item()
    test_sentiment_macro_f1 = test_sentiment_f1_macro.compute().item()
    test_emotion_weighted_f1 = test_emotion_f1_weighted.compute().item()
    test_sentiment_weighted_f1 = test_sentiment_f1_weighted.compute().item()

    # Reset metrics
    test_emotion_f1_macro.reset()
    test_sentiment_f1_macro.reset()
    test_emotion_f1_weighted.reset()
    test_sentiment_f1_weighted.reset()

    # Log the metrics
    logger.info(
        'Test: Epoch %d/%d | '
        'Loss: %.4f | '
        'Emotion Accuracy: %.4f | '
        'Sentiment Accuracy: %.4f | '
        'Emotion Macro F1: %.4f | '
        'Sentiment Macro F1: %.4f | '
        'Emotion Weighted F1: %.4f | '
        'Sentiment Weighted F1: %.4f',
        epoch + 1, epochs,
        epoch_test_loss,
        epoch_test_emotion_accuracy,
        epoch_test_sentiment_accuracy,
        test_emotion_macro_f1,
        test_sentiment_macro_f1,
        test_emotion_weighted_f1,
        test_sentiment_weighted_f1,
    )

    df_test = pd.DataFrame({
        'epoch': [epochs],
        'type': ['test'],
        'loss': [epoch_test_loss],
        'emotion_accuracy': [epoch_test_emotion_accuracy],
        'sentiment_accuracy': [epoch_test_sentiment_accuracy],
        'emotion_macro_f1': [test_emotion_macro_f1],
        'sentiment_macro_f1': [test_sentiment_macro_f1],
        'emotion_weighted_f1': [test_emotion_weighted_f1],
        'sentiment_weighted_f1': [test_sentiment_weighted_f1],
    })
    df = pd.concat([df, df_test])

    # Compute and save confusion matrices
    cm = {
        'emotion': test_emotion_cm.compute().cpu().numpy(),
        'sentiment': test_sentiment_cm.compute().cpu().numpy(),
    }

    return df, cm
