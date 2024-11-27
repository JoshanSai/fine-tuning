from transformers import pipeline


def run_pre_trained_model(question: str, context: str):
    # Load the pre-trained model and tokenizer
    qa_pipeline_pretrained = pipeline("question-answering", model="roberta-base", tokenizer="roberta-base")

    response_pretrained = qa_pipeline_pretrained(question=question, context=context)

    print("Response of pre-trained model:", response_pretrained)


def run_trained_model(question: str, context: str):
    # Load the pre-trained model and tokenizer
    qa_pipeline_pretrained = pipeline("question-answering", model="./fine_tuned_roberta",
                                      tokenizer="./fine_tuned_roberta")

    response_pretrained = qa_pipeline_pretrained(question=question, context=context)

    print("Response from model trained on squad:", response_pretrained)


def compare_models(question: str, context: str):
    run_pre_trained_model(question=question, context=context)
    run_trained_model(question=question, context=context)


context = """Mount Everest is the highest mountain in the world. It is located in the Himalayas on the border between Nepal and China. Its peak rises to 8,848.86 meters (29,031.7 feet) above sea level."""
question = "Where is Mount Everest located?"

compare_models(question=question, context=context)
