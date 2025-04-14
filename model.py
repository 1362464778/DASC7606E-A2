from transformers import AutoModelForTokenClassification

from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT


def initialize_model():
    """
    Initialize a model for token classification.

    Returns:
        A model for token classification.
    """
    # Initialize the model using the pre-trained checkpoint specified in constants.py
    # Configure it with the appropriate labels for NER token classification
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    
    return model