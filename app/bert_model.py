import torch
from torch import nn
from transformers import BertModel


class BertlNet(nn.Module):
    """
    A multi-label classification model using pre-trained BERT for textual data
    and additional speaker features.

    :param bert_model_name: Name of the pre-trained BERT model.
    :param speaker_feature_dim: Dimensionality of the speaker features.
    :param n_emotion_classes: Number of emotion classes.
    :param n_sentiment_classes: Number of sentiment classes.
    :param dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        bert_model_name: str,
        n_speakers: int,
        n_emotion_classes: int,
        n_sentiment_classes: int,
        dropout_rate: float,
    ):
        super().__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        # Speaker flow
        self.speaker_fc = nn.Sequential(
            nn.Linear(n_speakers, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        combined_size = hidden_size + (hidden_size // 4)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Classification heads
        self.emotion_classifier = nn.Linear(combined_size, n_emotion_classes)
        self.sentiment_classifier = nn.Linear(combined_size, n_sentiment_classes)

    def forward(
            self,
            x_utterance: torch.Tensor,
            attention_mask: torch.Tensor,
            x_speaker: torch.Tensor,
    ):
        """ Forward pass of the model."""
        # Utterance embeddings
        # print(f'{x_utterance.size()=}')
        # print(f'{attention_mask.size()=}')
        x_utterance = self.bert(
            input_ids=x_utterance,
            attention_mask=attention_mask,
        ).pooler_output
        # print(f'{x_utterance.size()=}')

        # Speaker flow
        # print(f'{x_speaker.size()=}')
        x_speaker = self.speaker_fc(x_speaker)
        # print(f'{x_speaker.size()=}')

        x = torch.cat((x_utterance, x_speaker), dim=1)
        # print(f'{x.size()=}')
        x = self.dropout(x)

        # Classification
        out_emotion = self.emotion_classifier(x)
        out_sentiment = self.sentiment_classifier(x)

        return out_emotion, out_sentiment
