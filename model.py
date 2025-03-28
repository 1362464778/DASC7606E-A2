from transformers import AutoModelForTokenClassification
from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT

def initialize_model():
    """
    Initialize a model for token classification.

    Returns:
        A model for token classification.
    """
    # 初始化预训练模型，用于 token classification
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        id2label=ID_TO_LABEL,  # 映射 ID 到标签
        label2id=LABEL_TO_ID  # 映射标签到 ID
    )

    return model