import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.logging_config import logger_config

logger = logging.getLogger(__name__)
logger_config(logger)


def visualisation(
        df: pd.DataFrame,
        cm: dict,
        labels: list,
        output_dir: Path,
):
    """
    Visualise confusion matrices and metrics, and save all results.

    :param df: DataFrame containing training, validation, and test metrics.
    :param cm: Dictionary containing confusion matrices for emotion and sentiment.
    :param labels: List of label names (e.g., ['Emotion', 'Sentiment'])
    :param output_dir: Directory where results will be saved.
    """
    # Create output directory
    if not output_dir.exists():
        output_dir.mkdir()

    # Save the DataFrame to CSV
    df.to_csv(output_dir / 'training_results.csv', index=False)

    # Separate datasets
    df_train = df.loc[df.loc[:, 'type'] == 'train', :]
    df_val = df.loc[df.loc[:, 'type'] == 'val', :]

    # Plot metrics and loss for each label
    for label in labels:
        metrics = [
            f'{label.lower()}_accuracy',
            f'{label.lower()}_precision',
            f'{label.lower()}_recall',
            f'{label.lower()}_macro_f1',
            f'{label.lower()}_weighted_f1',
        ]

        fig, ax1 = plt.subplots(figsize=(20, 12))

        # Plot the loss on the primary y-axis
        color_loss = 'tab:olive'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(
            df_train.loc[:, 'epoch'],
            df_train.loc[:, 'loss'],
            label='Train Loss',
            color=color_loss,
            linestyle='-',
        )
        ax1.plot(
            df_val.loc[:, 'epoch'],
            df_val.loc[:, 'loss'],
            label='Validation Loss',
            color=color_loss,
            linestyle='--',
        )
        ax1.tick_params(axis='y')

        # Create secondary y-axis for metrics
        ax2 = ax1.twinx()

        # Get default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, metric in enumerate(metrics):
            color = colors[i % len(colors)]
            # Plot train metric
            ax2.plot(
                df_train.loc[:, 'epoch'],
                df_train.loc[:, metric],
                label=f'Train {metric.replace(f"{label.lower()}_", "").title()}',
                color=color,
                linestyle='-',
            )
            # Plot validation metric
            ax2.plot(
                df_val.loc[:, 'epoch'],
                df_val.loc[:, metric],
                label=f'Validation {metric.replace(f"{label.lower()}_", "").title()}',
                color=color,
                linestyle='--',
            )
        ax2.set_ylabel('Metrics')
        ax2.tick_params(axis='y')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
        )

        plt.title(f'{label} Loss and Metrics over Epochs')
        plt.grid(True)
        fig.tight_layout()
        plt.savefig(
            output_dir / f'{label.lower()}_loss_metrics_over_epochs.png',
            bbox_inches='tight',
        )
        plt.close(fig)

    # Confusion Matrices
    for label in labels:
        cmatrix = cm[label.lower()]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{label} Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig(output_dir / f'{label.lower()}_confusion_matrix.png')
        plt.close()

    logger.info('All results have been saved to %s', output_dir)
