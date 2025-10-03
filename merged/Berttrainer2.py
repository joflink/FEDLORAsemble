import os
import json
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import Dataset
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AlbertPreTrainedModel,
    AlbertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# --- MODELLKONFIGURATION ---
MODEL_NAME = "albert-base-v1"
# Antal unika värden för varje klassificeringsuppgift
NUM_LABELS_CONFIG = {
    "category": 5,   # Ändra detta till ditt antal kategorier
    "difficulty": 2, # easy/hard
    "privacy": 2,    # true/false
}

# ==============================================================================
# DEL 1: MULTI-TASK MODELLKLASS (används endast för 'multitask'-läget)
# ==============================================================================
class AlbertForMultiTaskClassification(AlbertPreTrainedModel):
    """
    En anpassad Albert-modell med tre separata klassificeringshuvuden.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_category = config.num_labels_category
        self.num_labels_difficulty = config.num_labels_difficulty
        self.num_labels_privacy = config.num_labels_privacy

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Skapa ett klassificeringshuvud för varje uppgift
        self.classifier_category = nn.Linear(config.hidden_size, self.num_labels_category)
        self.classifier_difficulty = nn.Linear(config.hidden_size, self.num_labels_difficulty)
        self.classifier_privacy = nn.Linear(config.hidden_size, self.num_labels_privacy)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None, # Trainer skickar in alla label-kolumner som en enda tensor här
        **kwargs,
    ):
        # Hämta output från basmodellen
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # Få logits från varje huvud
        logits_category = self.classifier_category(pooled_output)
        logits_difficulty = self.classifier_difficulty(pooled_output)
        logits_privacy = self.classifier_privacy(pooled_output)

        # Beräkna total loss om labels finns
        total_loss = 0
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            # Labels kommer som en tensor [batch_size, 3]. Vi delar upp den.
            labels_category = labels[:, 0]
            labels_difficulty = labels[:, 1]
            labels_privacy = labels[:, 2]

            loss_category = loss_fct(logits_category, labels_category)
            loss_difficulty = loss_fct(logits_difficulty, labels_difficulty)
            loss_privacy = loss_fct(logits_privacy, labels_privacy)
            
            # Vikta gärna dessa om en uppgift är viktigare än de andra
            total_loss = loss_category + loss_difficulty + loss_privacy

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=(logits_category, logits_difficulty, logits_privacy),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ==============================================================================
# DEL 2: HJÄLPFUNKTIONER
# ==============================================================================

def load_and_prepare_dataset(tokenizer, task_type='multitask', target_label=None):
    """Laddar och förbereder datasetet för antingen multitask eller separat träning."""
    with open("trainingdata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples["question"], truncation=True, padding="max_length")
        if task_type == 'multitask':
            # Packa alla labels i en lista för multitask
            tokenized_inputs["labels"] = [
                list(label) for label in zip(
                    examples["category_label"],
                    examples["difficulty_label"],
                    examples["privacy_label"]
                )
            ]
        else:
            # Använd endast en specifik label för separat träning
            tokenized_inputs["labels"] = examples[target_label]
        return tokenized_inputs

    dataset = dataset.map(preprocess_function, batched=True)
    # Ta bort textkolumner som modellen inte behöver
    dataset = dataset.remove_columns(['question', 'category_label', 'difficulty_label', 'privacy_label'])
    
    return dataset.train_test_split(test_size=0.1)

# ==============================================================================
# DEL 3: TRÄNINGSFUNKTIONER
# ==============================================================================

def train_separate_models(tokenizer):
    """Tränar tre oberoende modeller, en för varje uppgift."""
    print("--- Startar träning av SEPARATA modeller ---")
    
    tasks = ["category", "difficulty", "privacy"]
    
    for task in tasks:
        print(f"\n--- Tränar modell för: {task.upper()} ---")
        
        target_label_col = f"{task}_label"
        num_labels = NUM_LABELS_CONFIG[task]
        output_dir = f"./router_{task}"

        # Ladda dataset för den specifika uppgiften
        split_dataset = load_and_prepare_dataset(tokenizer, task_type='separate', target_label=target_label_col)
        
        # Ladda standardmodellen med rätt antal labels
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=50, # Färre epoker kan räcka för enklare uppgifter
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir=f"./logs_{task}",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": accuracy_score(labels, preds)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()
        
        # Spara den bästa modellen
        print(f"Sparar bästa modell och tokenizer för {task} till {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

def train_multitask_model(tokenizer):
    """Tränar en enda modell som hanterar alla tre uppgifterna."""
    print("--- Startar träning av SAMMANSLAGEN (Multi-Task) modell ---")
    
    output_dir = "./router_multitask"

    # Ladda dataset för multitask
    split_dataset = load_and_prepare_dataset(tokenizer, task_type='multitask')

    # Skapa en config och ladda den anpassade modellen
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels_category = NUM_LABELS_CONFIG["category"]
    config.num_labels_difficulty = NUM_LABELS_CONFIG["difficulty"]
    config.num_labels_privacy = NUM_LABELS_CONFIG["privacy"]
    
    model = AlbertForMultiTaskClassification.from_pretrained(MODEL_NAME, config=config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100, # Kan behöva fler epoker för att lära sig alla uppgifter
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs_multitask",
        load_best_model_at_end=True,
        metric_for_best_model="eval_category_accuracy", # Välj en primär metric att optimera för
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    def compute_multitask_metrics(eval_pred):
        logits_tuple, labels = eval_pred
        logits_category, logits_difficulty, logits_privacy = logits_tuple

        preds_category = np.argmax(logits_category, axis=-1)
        preds_difficulty = np.argmax(logits_difficulty, axis=-1)
        preds_privacy = np.argmax(logits_privacy, axis=-1)
        
        # Packa upp labels från [batch_size, 3] tensorn
        labels_category = labels[:, 0]
        labels_difficulty = labels[:, 1]
        labels_privacy = labels[:, 2]

        return {
            "category_accuracy": accuracy_score(labels_category, preds_category),
            "difficulty_accuracy": accuracy_score(labels_difficulty, preds_difficulty),
            "privacy_accuracy": accuracy_score(labels_privacy, preds_privacy),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        compute_metrics=compute_multitask_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    print(f"Sparar bästa multitask-modell och tokenizer till {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# ==============================================================================
# DEL 4: HUVUDBLOCK FÖR ATT KÖRA SKRIPTET
# ==============================================================================

if __name__ == "__main__":
    # --- VÄLJ DITT TRÄNINGSLÄGE HÄR ---
    # Alternativ: "separate" eller "multitask"
    TRAINING_MODE = "separate" 
    
    # Ladda tokenizer en gång
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if TRAINING_MODE == "separate":
        train_separate_models(tokenizer)
    elif TRAINING_MODE == "multitask":
        train_multitask_model(tokenizer)
    else:
        raise ValueError("Ogiltigt TRAINING_MODE. Välj 'separate' eller 'multitask'.")

    print("\nTräningen är klar!")