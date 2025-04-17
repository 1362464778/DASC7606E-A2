from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


# def create_training_arguments() -> TrainingArguments:
#     """
#     Create and return the training arguments for the model.

#     Returns:
#         Training arguments for the model.

#     NOTE: You can change the training arguments as needed.
#     # Below is an example of how to create training arguments. You are free to change this.
#     # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
#     """
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         overwrite_output_dir=True,
#         do_train=False,
#         do_eval=True,
#         learning_rate=2e-5,
#         warmup_ratio=0.1,
#         load_best_model_at_end=True,
#         push_to_hub=False,
#         eval_strategy="steps",
#         fp16=True,
#     )

#     return training_args

def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=3e-5,              # Adjusted learning rate
        per_device_train_batch_size=32,  # Increased batch size
        per_device_eval_batch_size=32,
        num_train_epochs=4,              # Increased number of epochs
        weight_decay=0.02,               # Adjusted weight decay
        warmup_ratio=0.1,                # Added warmup ratio
        load_best_model_at_end=True,
        eval_strategy="steps",           # Evaluate at specific steps
        eval_steps=500,                  # Evaluate every 500 steps
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=2,              # Keep only the 2 best checkpoints
        metric_for_best_model="f1",      # Use F1 score to determine best model
        greater_is_better=True,          # Higher F1 score is better
        fp16=True,                       # Use mixed precision training if GPU supports it
        push_to_hub=False,
        resume_from_checkpoint=False,  # Resume from checkpoint if available
        eval_accumulation_steps=100
    )

    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for token classification.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.
    """
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args: TrainingArguments = create_training_arguments()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
