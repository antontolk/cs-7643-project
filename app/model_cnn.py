import torch
from torch import nn


class CNN1DNet(nn.Module):
    """The CNN 1D model for the emotion and sentiment predictions."""
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            n_speakers: int,
            n_classes: list,
            kernel_sizes: list,
            num_filters: list,
            hidden_size: int,
            dropout_rate: float,
            labels: list,
            use_batch_norm: bool
    ):
        """
        :param vocab_size: size of the vocabulary for embeddings
        :param embedding_dim: dimension of word embeddings
        :param n_classes: number of classes for each label
        :param kernel_sizes: list of kernel sizes for convolutions
        :param num_filters: number of filters for each convolutional layer
        :param hidden_size: the hidden size
        :param dropout_rate: dropout rate for regularization
        :param labels: labels to classify
        :param use_batch_norm: whether to use batch normalization.
        """
        super().__init__()

        self.labels = labels
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        #######################################################################
        # Utterance flow
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN layers
        conv_layers = []
        in_channels = embedding_dim
        for i, kernel_size in enumerate(kernel_sizes):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters[i],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            conv_layers.append(conv)
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(num_filters[i]))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout_rate))
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = num_filters[i]

        self.conv_sequence = nn.Sequential(*conv_layers)

        #######################################################################
        # Speaker flow
        self.linear_speaker = nn.Sequential(
            nn.Linear(n_speakers, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        #######################################################################
        # Output heads with Dropout
        # Emotion Head
        if 'Emotion' in labels:
            self.emotion_head = nn.Sequential(
                nn.Linear(
                    num_filters[-1] + hidden_size // 4,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, n_classes[labels.index('Emotion')])
            )

        # Sentiment Head
        if 'Sentiment' in labels:
            self.sentiment_head = nn.Sequential(
                nn.Linear(
                    num_filters[-1] + hidden_size // 4,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, n_classes[labels.index('Sentiment')])
            )

    def forward(self, x_utterance, x_speaker):
        """Network forward pass"""
        # Utterance flow
        x_utterance = self.embedding(x_utterance).permute(0, 2, 1)  # Batch, Embedding_dim, Seq_len
        x_utterance = self.conv_sequence(x_utterance)

        # If after pooling the sequence length is 1, squeeze the last dimension
        if x_utterance.shape[2] == 1:
            x_utterance = x_utterance.squeeze(-1)
        # Perform max over the remaining sequence length
        else:
            x_utterance, _ = torch.max(x_utterance, dim=2)

        # Speaker flow
        x_speaker = self.linear_speaker(x_speaker)

        # Concat Utterance and Speaker flows
        x = torch.cat((x_utterance, x_speaker), dim=1)

        # Output heads
        out_emotion = self.emotion_head(x) \
            if hasattr(self, 'emotion_head') else None
        out_sentiment = self.sentiment_head(x) \
            if hasattr(self, 'sentiment_head') else None

        return out_emotion, out_sentiment
