from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerNet(nn.Module):
  def __init__(self, vocab_size, n_features, hidden, labels, n_classes, 
               nhead=4, num_layers=2, max_len=512):
    
    super().__init__()
    
    self.embedding = nn.Embedding(vocab_size, n_features)
    
    if n_features % nhead != 0:
      adjusted_n_features = (n_features // nhead + 1) * nhead
      self.feature_projection = nn.Linear(n_features, adjusted_n_features)
      self.n_features = adjusted_n_features
    else:
      self.feature_projection = nn.Identity()
      self.n_features = n_features
    
    self.positional_encoding = PositionalEncoding(n_features, max_len=max_len)
    
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=n_features,
      nhead=nhead,
      dim_feedforward=hidden,
      activation='relu',
      batch_first=True
    )
    
    self.transformer_encoder = nn.TransformerEncoder(
      self.encoder_layer, num_layers=num_layers
    )
    
    self.labels = labels
    self.n_classes = n_classes
    
    if 'Emotion' in labels:
      self.emotion_head = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes[labels.index('Emotion')])        
      )
      
    if 'Sentiment' in labels:
      self.sentiment_head = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_classes[labels.index('Sentiment')])
      )
      
  def forward(self, x):
    x = x.long()
    x = self.embedding(x)
    x = self.feature_projection(x)
    
    x = self.positional_encoding(x)
    
    x = self.transformer_encoder(x)
    
    pooled_output = x[:, 0, :]
    
    out_emotion = self.emotion_head(pooled_output) if 'Emotion' in self.labels else None
    out_sentiment = self.sentiment_head(pooled_output) if 'Sentiment' in self.labels else None
    
    return out_emotion, out_sentiment
  
class PositionalEncoding(nn.Module):
  def __init__(self, n_features, max_len=512):
    super().__init__()
    
    self.max_len = max_len
    self.encoding = torch.zeros(max_len, n_features)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, n_features, 2) * (-math.log(10000.0) / n_features))
    
    # even
    self.encoding[:, 0::2] = torch.sin(position * div_term)
    if n_features % 2 == 1: #if odd
      self.encoding[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
      self.encoding[:, 1::2] = torch.cos(position * div_term)
    
    self.encoding = self.encoding.unsqueeze(0)
    
  def forward(self, x):
    seq_len = x.size(1)
    return x + self.encoding[:, :seq_len, :].to(x.device)