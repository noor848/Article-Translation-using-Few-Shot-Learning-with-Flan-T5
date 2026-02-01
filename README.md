# Article-Translation-using-Few-Shot-Learning-with-Flan-T5



# Article Translation using Few-Shot Learning with Flan-T5

A machine learning project that demonstrates English-to-Spanish translation using Google's Flan-T5 language model with few-shot learning techniques.

## Description

This project implements an article translation system that translates English news articles to Spanish using the Flan-T5-large model. Instead of fine-tuning, it leverages few-shot learning by providing the model with example translations to guide its output. The system uses articles from the CNN/DailyMail dataset and demonstrates how providing context examples can improve translation quality.

## Features

- **Few-Shot Learning**: Uses 2-3 example translations to guide the model
- **Flan-T5 Model**: Leverages Google's instruction-tuned T5 model for translation
- **CNN/DailyMail Dataset**: Works with real news articles for practical translation scenarios
- **No Fine-Tuning Required**: Achieves translations through prompt engineering alone
- **Customizable Prompts**: Easy to modify examples and adjust translation quality

## Requirements

```
transformers
tensorflow
datasets
```

## Installation

```bash
pip install transformers tensorflow datasets
```

## Usage

1. **Load the Model and Tokenizer**
```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", from_pt=True)
```

2. **Prepare Few-Shot Examples**
   - Select 2-3 high-quality article-translation pairs
   - Create a prompt that includes these examples

3. **Generate Translation**
```python
inputs = tokenizer(prompt_with_new_article, return_tensors="tf")
outputs = model.generate(inputs["input_ids"], max_length=500)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## How It Works

1. **Dataset Loading**: Loads articles from the CNN/DailyMail dataset
2. **Example Selection**: Three manually translated articles serve as examples
3. **Prompt Construction**: Builds a few-shot prompt with example pairs
4. **Translation**: Feeds new articles through the model with examples as context
5. **Evaluation**: Outputs can be reviewed and the prompt refined iteratively



## Contact

For questions or feedback, please open an issue on the project repository.
