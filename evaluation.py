import evaluate
import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from constants import ID_TO_LABEL

metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    """
    Compute evaluation metrics (precision, recall, f1) for predictions.

    First takes the argmax of the logits to convert them to predictions.
    Then we have to convert both labels and predictions from integers to strings.
    We remove all the values where the label is -100, then pass the results to the metric.compute() method.
    Finally, we return the overall precision, recall, and f1 score.

    Args:
        eval_predictions: Evaluation predictions.

    Returns:
        Dictionary with evaluation metrics. Keys: precision, recall, f1.
    """
    # Use torch.no_grad() to disable gradient calculation and save memory
    with torch.no_grad():
        # Extract logits and labels from EvalPrediction
        logits, labels = eval_predictions.predictions, eval_predictions.label_ids
        
        # Convert logits to predictions (take argmax along last dimension)
        predictions = np.argmax(logits, axis=-1)
        
        # Initialize lists to store true and predicted labels after processing
        true_labels = []
        predicted_labels = []
        
        # Process each sentence
        for prediction, label in zip(predictions, labels):
            true_sentence = []
            pred_sentence = []
            
            # Process each token in the sentence
            for pred_token, true_token in zip(prediction, label):
                # Skip tokens with label -100 (special tokens)
                if true_token != -100:
                    # Convert numeric labels to string labels using ID_TO_LABEL mapping
                    true_sentence.append(ID_TO_LABEL[true_token])
                    pred_sentence.append(ID_TO_LABEL[pred_token])
            
            # Add processed sentences to the lists
            true_labels.append(true_sentence)
            predicted_labels.append(pred_sentence)
    
    # Compute metrics using the seqeval evaluator
    # Set zero_division=0 to avoid warnings when there are no true samples for a class
    results = metric_evaluator.compute(
        predictions=predicted_labels,
        references=true_labels,
        zero_division=0
    )
    
    # Return the overall metrics
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }