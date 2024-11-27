# Roberta Question Answering Fine-Tuning

This repository demonstrates the fine-tuning of the RoBERTa model for the Question Answering (QA) task using the SQuAD dataset. The project includes two main functionalities:
1. **Fine-tuning a RoBERTa model** on the SQuAD dataset.
2. **Running both the pre-trained and fine-tuned models** to answer questions based on a given context.

## Prerequisites

Ensure that the following dependencies are installed before running the code. This project was developed using Python 3.9:

- `transformers`
- `datasets`
- `torch`
- `pandas`

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Training the Model
1. **Preparing the Dataset**:
The SQuAD dataset (Stanford Question Answering Dataset) is used for training and evaluating the model. It contains paragraphs of text, with each paragraph accompanied by several questions. The task is to predict the span of text that answers each question.

2. **Fine-Tuning the Model**:
You can set up the data fine-tune the RoBERTa model on the SQuAD dataset. To begin the training process, execute the following command in your terminal:
```commandline
python main.py
```

This command will start the training process and will fine-tune the pre-trained RoBERTa model on the SQuAD dataset. The script will automatically download the SQuAD dataset and start training the model using the dataset. Make sure that you have the necessary computational resources (such as a GPU) for efficient training.

3. **Evaluating the Model**:
After the model has been trained, you can evaluate the performance of the pre-trained and fine-tuned models. To do so, run the following command:
```commandline
python testing.py
```

This command will load both the pre-trained and fine-tuned models, evaluate them on a sample question-answering task, and print the results.

## Example Output
Upon running the testing command, you will see output for question:

**Where is Mount Everest located?**
```bash
Response of pre-trained model: {'score': 0.0007548289722763002, 'start': 164, 'end': 187, 'answer': '7 feet) above sea level'}
Response from model trained on SQuAD: {'score': 0.03948439657688141, 'start': 69, 'end': 82, 'answer': 'the Himalayas'}
```

## Explanation of Output
**Pre-trained Model Response**

Score: 0.0007548289722763002 — Indicates the confidence level of the model's answer. A low score suggests the pre-trained model's answer might not be very accurate.
Start: 164 — The starting position of the answer within the context.
End: 187 — The ending position of the answer within the context.
Answer: '7 feet) above sea level' — The answer extracted by the pre-trained model.

**Fine-tuned Model Response**

Score: 0.03948439657688141 — Represents the improved confidence after fine-tuning on the SQuAD dataset.
Start: 69 — The starting position of the answer after fine-tuning.
End: 82 — The ending position of the answer after fine-tuning.
Answer: 'the Himalayas' — The more accurate answer produced by the fine-tuned model.


## CONCLUSION
The fine-tuned model demonstrates a significant improvement in both accuracy and confidence when answering questions based on the provided context, highlighting the benefits of fine-tuning a pre-trained model on task-specific data like SQuAD.
