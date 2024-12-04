"""The model training functions"""
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, BertTokenizerFast
from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)


def bert_model_training(
        model: nn.Module,
        dl_train: DataLoader,
        dl_val: DataLoader,
        dl_test: DataLoader,
        epochs: int,
        criterion_type: str,
        lr: float,
        optimiser_val: str, #either Adam or AdamW,
        emotion_weights,
        sentiment_weights
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

    # Define the loss functions
    if criterion_type == 'ce':
        criterion_sentiment = nn.CrossEntropyLoss()
        criterion_emotion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Not supported criterion type.')

    # Define optimizer
    if optimiser_val=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimiser_val=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        train_emotion_correct, train_sentiment_correct,train_total = 0,0,0
        total_emotion_loss, total_sentiment_loss = 0, 0
    
        # Empty list to save model predictions
        total_preds = []

        # Training loop
        # for batch_X, batch_y in dl_train:
        for step, batch in enumerate(dl_train):
            batch = [r.to(device) for r in batch]
            # Unpack 3 elements: sent_id, mask, emotion_sentiment_labels (which is a 2D tensor)
            sent_id, mask, emotion_sentiment_labels = batch  # Emotion and sentiment labels together
            # Extract emotion and sentiment labels from the batch
            emotion_labels = emotion_sentiment_labels[:, 0]  # First column is emotion
            sentiment_labels = emotion_sentiment_labels[:, 1]  # Second column is sentiment

            # Clear previously calculated gradients
            model.zero_grad()
            # Get model predictions for the current batch
            out_emotion, out_sentiment = model(sent_id, mask)  # Predictions from all heads (emotion, sentiment)

            # Calculate individual task losses (Emotion & Sentiment)
            loss_emotion = criterion_emotion(out_emotion, emotion_labels)  # Emotion head loss
            loss_sentiment = criterion_sentiment(out_sentiment, sentiment_labels)  # Sentiment head loss loss

            # Total loss is the sum of both head losses
            total_epoch_loss = loss_emotion + loss_sentiment
            total_loss += loss_emotion.item() + loss_sentiment.item()
    
            # Add on to the total losses
            total_emotion_loss += loss_emotion.item()
            total_sentiment_loss += loss_sentiment.item()
            
            # Backward pass to calculate the gradients
            total_epoch_loss.backward()
    
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optimizer.step()
    
            total_preds.append((out_emotion, out_sentiment))


            # Accuracy Emotions
            _, y_pred_emotion = torch.max(out_emotion, 1)
            train_emotion_correct += (y_pred_emotion == emotion_labels).sum().item()
            train_total += emotion_labels.size(0)

            # Accuracy Sentiment
            _, y_pred_sentiment = torch.max(out_sentiment, 1)
            train_sentiment_correct += (y_pred_sentiment == sentiment_labels).sum().item()
            
        


        # Validation loop
        model.eval()
        total_val_loss, total_val_emotion_loss, total_val_sentiment_loss = 0, 0, 0
        total_val_accuracy = 0
        val_emotion_correct, val_sentiment_correct, val_total=0,0,0
        total_val_preds = []
    
        for step, batch in enumerate(dl_val):
            
            if step % 50 == 0 and not step == 0:
                print(f'  Batch {step} of {len(val_dataloader)}')
    
            # Push the batch to GPU
            batch = [r.to(device) for r in batch]
            sent_id, mask, emotion_sentiment_labels = batch  # Unpack 4 elements
            # Extract emotion and sentiment labels from the batch
            emotion_labels = emotion_sentiment_labels[:, 0]  # First column is emotion
            sentiment_labels = emotion_sentiment_labels[:, 1]  # Second column is sentiment
    
            
            # Deactivate autograd for evaluation
            with torch.no_grad():
                
                # Get model predictions for the current batch
                out_emotion, out_sentiment = model(sent_id, mask)
    
                # Calculate individual task losses (Emotion & Sentiment)
                loss_emotion = criterion_emotion(out_emotion, emotion_labels)  # Emotion head loss
                loss_sentiment = criterion_sentiment(out_sentiment, sentiment_labels)  # Sentiment head loss
    
                # Total loss is the sum of both head losses
                epoch_val_loss = loss_emotion + loss_sentiment
                total_val_loss += loss_emotion.item() + loss_sentiment.item()
    
                total_val_emotion_loss += loss_emotion.item()
                total_val_sentiment_loss += loss_sentiment.item()
                
                # Accuracy Emotions
                _, y_pred_emotion = torch.max(out_emotion, 1)
                val_emotion_correct += (y_pred_emotion == emotion_labels).sum().item()
                val_total += emotion_labels.size(0)

                # Accuracy Sentiment
                _, y_pred_sentiment = torch.max(out_sentiment, 1)
                val_sentiment_correct += (y_pred_sentiment == sentiment_labels).sum().item()
                
                total_val_preds.append((out_emotion, out_sentiment))
    
        # Average the losses over the dataset
        epoch_val_loss = total_val_loss / len(dl_val)
        epoch_val_emotion_loss = total_val_emotion_loss / len(dl_val)
        epoch_val_sentiment_loss = total_val_sentiment_loss / len(dl_val)
        epoch_val_emotion_accuracy = val_emotion_correct / val_total
        epoch_val_sentiment_accuracy = val_sentiment_correct / val_total

        # Average the losses over the dataset
        epoch_train_loss = total_loss / len(dl_train)
        epoch_train_emotion_loss = total_emotion_loss / len(dl_train)
        epoch_train_sentiment_loss = total_sentiment_loss / len(dl_train)
        
        epoch_train_emotion_accuracy = train_emotion_correct / train_total
        epoch_train_sentiment_accuracy = train_sentiment_correct / train_total
        
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
