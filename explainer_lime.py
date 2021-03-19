import argparse
import json
from pathlib import Path
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig
from pyvi import ViTokenizer

parser = argparse.ArgumentParser(description="topic")
parser.add_argument(
    "--config_path",
    default="data/config.json",
    help="path to model bert",
)
args = parser.parse_args()


class WrapedSenti:
    def __init__(self, args):
        model_name = "avichr/heBERT"
        self.config = BertConfig.from_pretrained(model_name, num_labels=3)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 128

    def predict(self, texts):
        input_ids = self.tokenizer.encode(
            text=texts,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            is_split_into_words=True
        )
        # print(input_ids)
        results = self.model(
            torch.tensor([input_ids])
        )
        # print(softmax(results[0], axis=0).argmax(axis=1))
        print(softmax(results[0].detach().numpy()))
        import numpy as np
        return softmax(results[0].detach().numpy())


def explainer(args, text, num_samples=20):
    """Run LIME explainer on provided classifier"""

    model = WrapedSenti(args)
    predictor = model.predict

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        # Specify split option
        split_expression=lambda x: x.split(),
        # Our classifer uses bigrams or contextual ordering to classify text
        # Hence, order matters, and we cannot use bag of words.
        bow=False,
        class_names=["trung tính", "tiêu cực", "tích cực"],
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=20,
        num_samples=num_samples,
    )
    return exp


if __name__ == "__main__":
    # texts = ['Đâm mẹ bạn gái trọng thương, kẻ máu lạnh lĩnh án 16 năm tù']
    texts = ['i love you, i like you']
    # texts = ["mọi chuyện diễn ra không như ngày thường"]
    # texts = ['bản ABS nhìn sang chảnh hoành tráng']
    for i, text in enumerate(texts):
        exp = explainer(args, text, 1)
        output_filename = Path(
            __file__
        ).parent / "outputs/{}-explanation-lime.html".format(i + 1)
        exp.save_to_file(output_filename)
