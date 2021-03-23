# Transformers Visualise

Transformers Interpret is a model explainability tool designed to work exclusively with the 🤗  [transformers][transformers] package.
Supported:
* Python >= 3.6 
* Pytorch >= 1.5.0 
* [transformers][transformers] >= v3.0.0 
* captum >= 0.3.1 

The package does not work with Python 2.7 or below.
## Install

```bash
sudo apt-get install python3-tk
pip install transformers-interpret
pip install lime
```
# Documentation for lime
* script explainer
```bash
python explainer_lime.py
```


# Documentation for interpret
* Script explainer
```bash
python explainer_viz.py
```
## Quick Start
Let's start by initializing a transformers' model and tokenizer, and running it through the `SequenceClassificationExplainer`.

For this example we are using `avichr/heBERT`, sentiment analysis task.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "avichr/heBERT"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# With both the model and tokenizer initialized we are now able to get explanations on an example text.
from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    "I love you, I like you", 
    model, 
    tokenizer)
attributions = cls_explainer()
```

Which will return the following list of tuples:

```python
attributions.word_attributions
# [('[CLS]', 0.0),
#  ('i', 0.2778544699186709),
#  ('love', 0.7792370723380415),
#  ('you', 0.38560088858031094),
#  (',', -0.01769750505546915),
#  ('i', 0.12071898121557832),
#  ('like', 0.19091105304734457),
#  ('you', 0.33994871536713467),
#  ('[SEP]', 0.0)]
```

Positive attribution numbers indicate a word contributes positively towards the predicted class, while negative numbers indicate a word contributes negatively towards the predicted class. Here we can see that **I love you** gets the most attention.
You can use `predicted_class_index` in case you'd want to know what the predicted class actually is:

```python
cls_explainer.predicted_class_name
# 'POSITIVE'
```


### Visualizing attributions

Sometimes the numeric attributions can be difficult to read particularly in instances where there is a lot of text. To help with that we also provide the `visualize()` method that utilizes Captum's in built viz library to create a HTML file highlighting the attributions.

If you are in a notebook, calls to the `visualize()` method will display the visualization in-line. Alternatively you can pass a filepath in as an argument and an HTML file will be created, allowing you to view the explanation HTML in your browser.

```python
cls_explainer.visualize("distilbert_viz.html")
```

### Explaining Attributions for Non Predicted Class

Attribution explanations are not limited to the predicted class. Let's test a more complex sentence that contains mixed sentiments.

In the example below we pass `class_name="NEGATIVE"` as an argument indicating we would like the attributions to be explained for the **NEGATIVE** class regardless of what the actual prediction is. Effectively because this is a binary classifier we are getting the inverse attributions.

```python
cls_explainer = SequenceClassificationExplainer("I love you, I like you, I also kinda dislike you", model, tokenizer)
attributions = cls_explainer(class_name="NEGATIVE")
```

In this case, `predicted_class_name` still returns a prediction of the **POSITIVE** class, because the model has generated the same prediction but nonetheless we are interested in looking at the attributions for the negative class regardless of the predicted result. 

```python
cls_explainer.predicted_class_name
# 'POSITIVE'
```

## Miscellaneous

**Captum Links**

Below are some links I used to help me get this package together using Captum. Thank you to @davidefiocco for your very insightful GIST.

- [Link to useful GIST on captum](https://gist.github.com/davidefiocco/3e1a0ed030792230a33c726c61f6b3a5)
- [Link to runnable colab of captum with BERT](https://colab.research.google.com/drive/1snFbxdVDtL3JEFW7GNfRs1PZKgNHfoNz)

[transformers]: https://huggingface.co/transformers/
