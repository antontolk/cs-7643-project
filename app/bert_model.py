import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert, labels,n_classes):
        super(BERT_Arch, self).__init__()
        
        self.bert = bert 
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
      
        # ReLU activation function
        self.relu = nn.ReLU()

        # Shared dense layer
        self.fc_shared = nn.Linear(768, 512)
        
        # Emotion head
        if 'Emotion' in labels:
            num_emotion_classes = n_classes[0]
            self.emotion_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_emotion_classes),  # Output layer for Emotion
                nn.LogSoftmax(dim=1)
            )

        # Sentiment head
        if 'Sentiment' in labels:
            num_sentiment_classes = n_classes[1]
            self.sentiment_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_sentiment_classes),  # Output layer for Sentiment
                nn.LogSoftmax(dim=1)
            )

    # Define the forward pass
    def forward(self, sent_id, mask):
        
        # Pass the inputs to the BERT model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        # Shared dense layer
        x = self.fc_shared(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        # Get outputs from both heads
        emotion_output = self.emotion_head(x)
        sentiment_output = self.sentiment_head(x)

        return emotion_output, sentiment_output