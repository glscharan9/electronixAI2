import argparse, json, random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def set_seed(seed):
    """Pins random seeds for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Fine-tune a sentiment analysis model.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.jsonl file.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    args = parser.parse_args()

    texts, labels = [], []
    label_map = {"positive": 1, "negative": 0, "neutral": 2}

    with open(args.data, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['label'] in label_map:
                texts.append(item['text'])
                labels.append(label_map[item['label']])

    # Using roberta-base model and tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    print("Saving fine-tuned model to ./model/")
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    main()