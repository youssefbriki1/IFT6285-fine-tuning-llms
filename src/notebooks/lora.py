# %% [markdown]
# # Fine-tuning DeepSeek-R1-Distill-Qwen-7B with LoRA
# Sentiment classification on FinancialPhraseBank (3 labels).
# LoRA adapts only a small subset of weights → faster + cheaper fine-tuning.

# %%
import os
import csv
import getpass

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv, find_dotenv
from datasets import Dataset, DatasetDict

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import LoraConfig, TaskType, get_peft_model
import wandb

# %% [markdown]
# ## Environment Setup

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SCRATCH = os.environ.get("SCRATCH", os.path.expanduser("~"))
HF_HOME = os.path.join(SCRATCH, "huggingface_cache")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

CACHE_DIR = HF_HOME
PROJECT_ROOT = os.getcwd()

RAW_DATA_DIR = (
    "/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/data/"
    "FinancialPhraseBank-v1.0"
)

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment_data")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "deepseek_fpbank_lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# %% [markdown]
# ## Prepare FinancialPhraseBank Dataset

# %%
def load_fpbank_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='@', engine='python',
                       names=['text', 'sentiment'], on_bad_lines='skip')


def prepare_fpbank_csv_splits() -> None:
    train_files = ["Sentences_75Agree_utf8.txt", "Sentences_AllAgree_utf8.txt"]
    test_file = "Sentences_50Agree_utf8.txt"

    splits = {
        "train": pd.concat(
            [load_fpbank_file(os.path.join(RAW_DATA_DIR, f)) for f in train_files],
            ignore_index=True,
        ),
        "test": load_fpbank_file(os.path.join(RAW_DATA_DIR, test_file)),
    }

    for name, df in splits.items():
        df = df[["sentiment", "text"]]
        out_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.csv")
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"{name}: {len(df)} examples → {out_path}")


prepare_fpbank_csv_splits()

# %% [markdown]
# ## Tokenization + Dataset Formatting

# %%
def build_datasets(tokenizer: AutoTokenizer) -> DatasetDict:
    df_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train, preserve_index=False),
        "test": Dataset.from_pandas(df_test, preserve_index=False),
    })

    def preprocess(batch):
        enc = tokenizer(
            list(batch["text"]),
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        enc["labels"] = [LABEL2ID[s] for s in batch["sentiment"]]
        return enc

    for split in ds:
        ds[split] = ds[split].map(
            preprocess,
            batched=True,
            batch_size=500,
            remove_columns=["text", "sentiment"],
            num_proc=4,
        )
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# %% [markdown]
# ## Model + LoRA Configuration

# %%
def build_peft_model(use_dora=False):
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=CACHE_DIR,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.01,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        use_dora=use_dora,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


model, tokenizer = build_peft_model(use_dora=False)

# %% [markdown]
# ## Build Dataset

# %%
ds = build_datasets(tokenizer)
ds

# %% [markdown]
# ## Training

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% [markdown]
# ## Save Model

# %%
trainer.save_model(os.path.join(OUTPUT_DIR, "model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

print("Training complete.")
