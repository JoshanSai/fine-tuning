import os
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, pipeline
from transformers import TrainingArguments, Trainer


class RobertaQuestionAnswering:
    def __init__(self, model_name="roberta-base", dataset_name="squad", output_dir="./results"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = RobertaForQuestionAnswering.from_pretrained(self.model_name)

    def load_and_preprocess_dataset(self):
        """
        Load and preprocess the dataset (e.g., SQuAD).
        """
        squad = load_dataset(self.dataset_name)
        squad["train"] = squad["train"].select(range(500))  # Use the first 1000 examples
        squad["validation"] = squad["validation"].select(range(100))  # Use the first 200 examples

        def preprocess_function(examples):
            questions = examples["question"]
            contexts = examples["context"]
            answers = examples["answers"]

            # Tokenize with truncation and padding
            tokenized_examples = self.tokenizer(
                questions,
                contexts,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_offsets_mapping=True,
            )

            start_positions = []
            end_positions = []
            for i, offset_mapping in enumerate(tokenized_examples["offset_mapping"]):
                answer = answers[i]["text"][0] if answers[i]["text"] else ""
                start_char = answers[i]["answer_start"][0] if answers[i]["answer_start"] else None
                end_char = start_char + len(answer) if start_char is not None else None

                if start_char is None or end_char is None or not answer:
                    # Set default positions for invalid answers
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                # Find token start index
                token_start_index = 0
                while (
                        token_start_index < len(offset_mapping)
                        and offset_mapping[token_start_index][0] <= start_char
                ):
                    token_start_index += 1

                token_end_index = token_start_index
                while (
                        token_end_index < len(offset_mapping)
                        and offset_mapping[token_end_index][1] < end_char
                ):
                    token_end_index += 1

                # Add start and end positions, ensuring valid ranges
                start_positions.append(max(0, token_start_index - 1))
                end_positions.append(max(0, token_end_index - 1))

            tokenized_examples["start_positions"] = start_positions
            tokenized_examples["end_positions"] = end_positions
            tokenized_examples.pop("offset_mapping")  # Remove offset mapping for model compatibility
            return tokenized_examples

        tokenized_squad = squad.map(preprocess_function, batched=True)
        return tokenized_squad

    def train_model(self, train_dataset, eval_dataset):
        """
        Fine-tune the Roberta model using the preprocessed dataset.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        results = trainer.evaluate()
        print("Evaluation Results:", results)

        # Save the fine-tuned model and tokenizer
        self.model.save_pretrained("./fine_tuned_roberta")
        self.tokenizer.save_pretrained("./fine_tuned_roberta")

    def create_and_run_pipeline(self, question: str, context: str, model_dir="./fine_tuned_roberta"):
        """
        Load the fine-tuned model and tokenizer into a QA pipeline.
        """
        qa_pipeline = pipeline("question-answering", model=model_dir, tokenizer=model_dir)
        result = qa_pipeline(question=question, context=context)
        return result


def main():
    # Initialize the QA trainer
    qa_trainer = RobertaQuestionAnswering()

    # Load and preprocess the dataset
    tokenized_data = qa_trainer.load_and_preprocess_dataset()

    # Fine-tune the model
    qa_trainer.train_model(tokenized_data["train"], tokenized_data["validation"])


if __name__ == "__main__":
    main()
