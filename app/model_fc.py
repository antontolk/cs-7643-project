import torch
from torch import nn


class FullyConnectedNet(nn.Module):
    """The fully connected model for the emotion and sentiment predictions."""
    def __init__(
            self,
            n_features: int,
            hidden: int,
            labels: list,
            n_classes: list,
    ):
        """
        :param n_features: the number of features
        :type n_features: int
        :param hidden: the hidden size
        :type hidden: int
        :param labels: Labels to be classified
        :param n_classes: the number of classes for each label
        :type n_classes: list
        """

        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        self.linear_1 = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU()
        )

        # Output heads
        # Emotion
        if 'Emotion' in labels:
            self.emotion_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes[labels.index('Emotion')]),
            )

        # Sentiments
        if 'Sentiment' in labels:
            self.sentiment_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes[labels.index('Sentiment')]),
            )

    def forward(self, x_utterance, x_speaker):
        """Network forward pass"""
        # Concat inputs
        x = torch.cat((x_utterance, x_speaker), dim=1)

        x = self.linear_1(x)
        out_emotion = self.emotion_head(x)
        out_sentiment = self.sentiment_head(x)

        return out_emotion, out_sentiment

