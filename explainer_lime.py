import argparse
import json
import numpy as np
from pathlib import Path
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax
from typing import List
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

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

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        encode_ids = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        print(encode_ids)
        results = self.model(
            input_ids=encode_ids.input_ids,
            attention_mask=encode_ids.attention_mask
        )
        print(results[0].detach().numpy())
        # print(softmax(results[0].detach().numpy()))
        return results[0].detach().numpy()


def explainer(args, text, num_samples: int = 20):
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
        exp = explainer(args, text, 10)
        output_filename = Path(
            __file__
        ).parent / "outputs/{}-explanation-lime.html".format(i + 1)
        exp.save_to_file(output_filename)
