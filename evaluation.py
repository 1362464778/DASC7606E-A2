import numpy as np
from transformers.trainer_utils import EvalPrediction
import evaluate
from constants import ID_TO_LABEL  # 从constants.py导入映射关系

metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    # 获取预测的类别索引 (shape: [batch_size, sequence_length])
    predictions = np.argmax(eval_predictions.predictions, axis=2)
    labels = eval_predictions.label_ids  # shape: [batch_size, sequence_length]

    # 过滤特殊标签-100，并将ID转换为字符串标签
    true_labels = []
    true_predictions = []

    # 遍历每个样本
    for label_seq, pred_seq in zip(labels, predictions):
        filtered_labels = []
        filtered_preds = []
        # 遍历序列中的每个token
        for label_id, pred_id in zip(label_seq, pred_seq):
            if label_id != -100:  # 过滤padding部分
                filtered_labels.append(ID_TO_LABEL[label_id])
                filtered_preds.append(ID_TO_LABEL[pred_id])
        true_labels.append(filtered_labels)
        true_predictions.append(filtered_preds)

    # 计算指标
    results = metric_evaluator.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }