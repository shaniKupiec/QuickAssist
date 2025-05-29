import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider


class IntentHandler:
    def __init__(self, model_config, main_config):
        self.model_config = model_config
        self.main_config = main_config
        self.device = main_config['default_device']
        self.model_type = model_config.get('type')
        self.setup_model()

    def setup_model(self):
        if self.model_type == "ground_truth":
            return

        elif self.model_type == "bert":
            output_dir = self.model_config.get('output_dir', '')
            fine_tuned = self.model_config.get('fine_tuned', False)

            if fine_tuned and os.path.isdir("results"):
                checkpoints = [d for d in os.listdir("results") if d.startswith("checkpoint")]
                if checkpoints:
                    latest_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    model_path = os.path.join("results", latest_ckpt)
                else:
                    model_path = self.model_config['name']
            else:
                model_path = self.model_config['name']

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        elif self.model_type == "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config['name']).to(self.device)

        # TODO: finish flow for gpt-4
        elif self.model_type == "gpt-4":
            self.model = Agent[None, str](
                model=GroqModel(
                    self.model_config['model_name'],
                    provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
                ),
                system_prompt="You are an intent classifier for customer service queries. Return only the intent label."
            )

    async def get_intent(self, query, ground_truth_intent=None):
        if self.model_type == "ground_truth":
            if ground_truth_intent is None:
                raise ValueError("Ground truth intent required but not provided")
            return ground_truth_intent

        elif self.model_type == "bert":
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_intent = self.model.config.id2label[outputs.logits.argmax().item()]
            return predicted_intent

        elif self.model_type == "t5":
            prompt = f"Classify the intent: {query}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)
            predicted_intent = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return predicted_intent.strip()

        # TODO: finish flow for gpt-4
        elif self.model_type == "gpt-4":
            result = await self.model.run(query)
            return result.output.strip()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, train_data, eval_data=None):
        if self.model_type != "bert":
            raise ValueError(f"Model type {self.model_type} is not trainable")

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(train_data["intent"])
        texts = train_data["input"].tolist()

        self.model.num_labels = len(label_encoder.classes_)
        self.model.config.label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
        self.model.config.id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


        train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
        train_encodings = {k: list(v) for k, v in train_encodings.items()} # ensure lists
        train_encodings["labels"] = list(y_train)
        train_dataset = Dataset.from_dict(train_encodings)

        # train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
        # train_encodings["labels"] = y_train.tolist()
        # train_dataset = Dataset.from_dict(train_encodings)

        test_encodings = self.tokenizer(X_test, truncation=True, padding=True)
        test_encodings = {k: list(v) for k, v in test_encodings.items()} # ensure lists
        test_encodings["labels"] = list(y_test)
        test_dataset = Dataset.from_dict(test_encodings)


        # test_encodings = self.tokenizer(X_test, truncation=True, padding=True)
        # test_encodings["labels"] = y_test.tolist()
        # test_dataset = Dataset.from_dict(test_encodings)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            acc = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
            return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

        training_args = TrainingArguments(
            output_dir=self.model_config["output_dir"],
            per_device_train_batch_size=self.main_config["batch_size"],
            num_train_epochs=self.main_config["epochs"],
            logging_dir="./logs",
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        try:
            trainer.train()
            trainer.save_model(self.model_config["output_dir"])
        except Exception as e:
            print("❌ Training failed:", e)

        return self.model
