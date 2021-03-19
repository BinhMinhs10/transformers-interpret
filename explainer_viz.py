#     "0": "natural",
#     "1": "positive",
#     "2": "negative"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

if __name__ == "__main__":

    model_name = "avichr/heBERT"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # With both the model and tokenizer initialized we are now able to get explanations on an example text.

    cls_explainer = SequenceClassificationExplainer(
        "I love you, I like you",
        model,
        tokenizer)
    attributions = cls_explainer()
    # print(attributions.predicted_class_name)

    cls_explainer.visualize("outputs/explainer-viz.html")
