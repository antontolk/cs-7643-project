import torch
from torch import nn


class FullyConnectedNet(nn.Module):
    """The fully connected model for the emotion and sentiment predictions."""
    def __init__(
            self,
            n_features: int,
            hidden_size: int,
            labels: list,
            n_classes: list,
            dropout_rate: float,
            use_batch_norm: bool,
    ):
        """
        :param n_features: the number of features
        :type n_features: int
        :param hidden_size: the hidden size
        :type hidden_size: int
        :param labels: Labels to be classified
        :param n_classes: the number of classes for each label
        :type n_classes: list
        :param dropout_rate: Dropout probability for regularization.
        :type dropout_rate: float
        :param use_batch_norm: Whether to use Batch Normalization.
        :type use_batch_norm: bool
        """

        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        # Input
        input_layers = [nn.Linear(n_features, hidden_size)]
        if use_batch_norm:
            input_layers.append(nn.BatchNorm1d(hidden_size))
        input_layers.append(nn.ReLU())
        input_layers.append(nn.Dropout(dropout_rate))
        self.linear_input = nn.Sequential(*input_layers)

        # Output: Emotion head
        if 'Emotion' in labels:
            emotion_layers = [nn.Linear(hidden_size, hidden_size)]
            if use_batch_norm:
                emotion_layers.append(nn.BatchNorm1d(hidden_size))
            emotion_layers.append(nn.ReLU())
            emotion_layers.append(nn.Dropout(dropout_rate))
            emotion_layers.append(nn.Linear(
                hidden_size,
                n_classes[labels.index('Emotion')],
            ))
            self.emotion_head = nn.Sequential(*emotion_layers)

        # Output: Sentiments head
        if 'Sentiment' in labels:
            sentiment_layers = [nn.Linear(hidden_size, hidden_size)]
            if use_batch_norm:
                sentiment_layers.append(nn.BatchNorm1d(hidden_size))
            sentiment_layers.append(nn.ReLU())
            sentiment_layers.append(nn.Dropout(dropout_rate))
            sentiment_layers.append(nn.Linear(hidden_size, n_classes[labels.index('Sentiment')]))
            self.sentiment_head = nn.Sequential(*sentiment_layers)

    def forward(self, x_utterance, x_speaker):
        """Network forward pass"""
        # Concat inputs
        x = torch.cat((x_utterance, x_speaker), dim=1)
        x = self.linear_input(x)

        # Pass through each head
        out_emotion = self.emotion_head(x) \
            if hasattr(self, 'emotion_head') else None
        out_sentiment = self.sentiment_head(x) \
            if hasattr(self, 'sentiment_head') else None

        return out_emotion, out_sentiment
