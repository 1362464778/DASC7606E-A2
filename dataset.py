from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def align_labels_with_tokens(
        labels: list[int],
        word_ids: list[int | None],
) -> list[int]:
    """
    Align labels with tokenized word IDs, ensuring special tokens are ignored (-100).
    """
    aligned_labels = []
    previous_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:  # Special token
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # New word: Use the label of the first token in the word
            aligned_labels.append(labels[word_idx])
        else:
            # Inside the word: Use the "I-" label
            aligned_labels.append(labels[word_idx] if labels[word_idx] > 0 else -100)

        previous_word_idx = word_idx

    return aligned_labels


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    Tokenize input examples and align labels for token classification.

    To preprocess our whole dataset, we need to tokenize all the inputs and apply align_labels_with_tokens() on all the labels.
    To take advantage of the speed of our fast tokenizer, it’s best to tokenize lots of texts at the same time,
    so this function will processes a list of examples and return a list of tokenized inputs with aligned labels.

    Args:
        examples: Input examples.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    aligned_labels = [
        align_labels_with_tokens(
            labels=labels,
            word_ids=tokenized_inputs.word_ids(i),
        )
        for i, labels in enumerate(all_labels)
    ]
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def build_dataset() -> DatasetDict:
    """
    Build the dataset.

    Returns:
        The dataset in the form of a DatasetDict.
    """
    # 例子：加载MultiCoNER数据集
    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')

    # 添加对测试数据集的引用
    raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")

    return raw_datasets


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets
