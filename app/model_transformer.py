import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements the standard sinusoidal positional encoding as described in the Transformer paper.
    Adds positional information to the input embeddings.
    """

    def __init__(
            self,
            d_model: int,
            max_len: int,
    ):
        """
        Initializes the PositionalEncoding module.

        :param d_model: The dimension of the embeddings. Must match the
            `d_model` parameter in the Transformer encoder.
        :param max_len: The maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()

        # Matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the positional encodings in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Register as a buffer to avoid updating during training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerNet(nn.Module):
    """Transformer-based model for emotion and sentiment predictions."""
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            n_speakers: int,
            n_classes: list,
            n_heads: int,
            n_layers: int,
            hidden_size: int,
            dropout_rate: float,
            labels: list,
    ):
        """
        Initialize the TransformerNet model with Transformer encoder layers and a [CLS] token.

        :param vocab_size: size of the vocabulary for embeddings.
        :param embedding_dim: dimension of word embeddings.
        :param n_speakers: number of speaker features.
        :param n_classes: Number of classes for each label.
        :param n_heads: Number of attention heads in the Transformer encoder.
        :param n_layers: Number of Transformer encoder layers.
        :param hidden_size: Dimension of the feedforward network in the Transformer encoder.
        :param dropout_rate: Dropout probability for regularization.
        :param labels: List of labels to classify (e.g., ['Emotion', 'Sentiment']).
        """
        super().__init__()

        self.labels = labels
        self.dropout_rate = dropout_rate

        # Embedding Layer with [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=100,
        )

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_size,
            dropout=dropout_rate,
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=n_layers,
        )

        # Speaker Flow
        self.linear_speaker = nn.Sequential(
            nn.Linear(n_speakers, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Output head: Emotion
        if 'Emotion' in labels:
            self.emotion_head = nn.Sequential(
                nn.Linear(
                    embedding_dim + hidden_size // 4,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, n_classes[labels.index('Emotion')])
            )

        # Output head: Sentiment
        if 'Sentiment' in labels:
            self.sentiment_head = nn.Sequential(
                nn.Linear(
                    embedding_dim + hidden_size // 4,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, n_classes[labels.index('Sentiment')])
            )

    def forward(self, x_utterance: torch.Tensor, x_speaker: torch.Tensor):
        """Network forward pass."""

        batch_size, seq_len = x_utterance.size()

        # Utterance flow
        x_utterance = self.embedding(x_utterance)

        # Expand [CLS] token to match batch size and prepend to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_utterance = torch.cat((cls_tokens, x_utterance), dim=1)

        # Add positional encoding
        x_utterance = self.pos_encoder(x_utterance).permute(1, 0, 2)

        # Pass through transformer encoder
        x_utterance = self.transformer_encoder(x_utterance).permute(1, 0, 2)

        # Extract the [CLS] token's embedding
        x_utterance = x_utterance[:, 0, :]

        # Speaker Flow
        x_speaker = self.linear_speaker(x_speaker)

        x = torch.cat((x_utterance, x_speaker), dim=1)

        # Output heads
        out_emotion = self.emotion_head(x) \
            if hasattr(self, 'emotion_head') else None
        out_sentiment = self.sentiment_head(x) \
            if hasattr(self, 'sentiment_head') else None

        return out_emotion, out_sentiment