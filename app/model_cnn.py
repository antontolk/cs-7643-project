import torch
from torch import nn


class CNN1DNet(nn.Module):
    """The CNN 1D model for the emotion and sentiment predictions."""
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            n_classes: list,
            kernel_sizes: list,
            num_filters: int,
            dropout: float,
            labels: list,
    ):
        """
        :param vocab_size: size of the vocabulary for embeddings
        :param embedding_dim: dimension of word embeddings
        :param n_classes: number of classes for each label
        :param kernel_sizes: list of kernel sizes for convolutions
        :param num_filters: number of filters for each convolutional layer
        :param dropout: dropout rate for regularization
        :param labels: labels to classify
        """
        super().__init__()

        self.labels = labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Output heads
        # Emotion
        if 'Emotion' in labels:
            self.emotion_head = nn.Sequential(
                nn.Linear(len(kernel_sizes) * num_filters, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes[labels.index('Emotion')])
            )
        
        # Sentiments
        if 'Sentiment' in labels:
            self.sentiment_head = nn.Sequential(
                nn.Linear(len(kernel_sizes) * num_filters, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes[labels.index('Sentiment')])
            )

    def forward(self, x):
        """Network forward pass"""
        x = self.embedding(x).permute(0, 2, 1)  # Batch, Embedding_dim, Seq_len

        conv_outs = [self.global_pool(nn.ReLU()(conv(x))).squeeze(-1)
                     for conv in self.conv_layers]
        x = self.dropout(torch.cat(conv_outs, dim=1))

        out_emotion = self.emotion_head(x)
        out_sentiment = self.sentiment_head(x)

        return out_emotion, out_sentiment
