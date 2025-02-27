import os
import json
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Load dataset
with open("trainingdata.json", "r") as f:
    data = json.load(f)

# Ensure "labels" exist
for example in data:
    example["labels"] = example["expert"]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Load tokenizer
model_name = "albert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["question"], truncation=True, padding="max_length")
    tokenized_inputs["labels"] = examples["labels"]
    return tokenized_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into train/eval
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5,   hidden_dropout_prob=0.1 )

# Define a metric function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Custom callback to save only every 500 epochs
class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_interval=500):
        self.save_interval = save_interval

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and int(state.epoch) % self.save_interval == 0 and state.epoch > 0:
            control.should_save = True
        else:
            control.should_save = False

# Training arguments (fixing save & eval strategy issue)
training_args = TrainingArguments(
    output_dir="./bert_router",
    save_strategy="epoch",
    save_steps=50,  # Save every 1000 training steps       # ✅ Ensures best model can be saved
    eval_strategy="epoch",           # ✅ Updated from `evaluation_strategy` (now deprecated)
    load_best_model_at_end=True,     # ✅ Loads best checkpoint
    metric_for_best_model="accuracy",
    greater_is_better=True,
    num_train_epochs=1000,          # Large number, but early stopping will prevent excessive training
    per_device_train_batch_size=20,
    per_device_eval_batch_size=5,
    save_total_limit=10,              # ✅ Prevents too many saved checkpoints
    logging_dir="./logs",
    report_to="none",
    weight_decay=0.01 , # ✅ Lägg till viktregulering
    learning_rate=5e-5,               # ✅ Lägre lärhastighet för mer stabil träning
    lr_scheduler_type="cosine",        # ✅ Långsamt minskande inlärningshastighet
    warmup_ratio=0.1                   # ✅ Värm upp de första 10% av stegen
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=10),  # Stop if no improvement in 5 epochs
        # SaveEveryNEpochsCallback(save_interval=500)        # Save only every 500 epochs
    ],
)

# Train the model
trainer.train()

# Save the final trained model (best version)
trainer.save_model("bert-router")
tokenizer.save_pretrained("bert-router")
