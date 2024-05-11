
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    NllbTokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer, NllbTokenizer]

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class FonTextToText:
    def __init__(self, tofrench_model_name="masakhane/afrimt5_fon_fr_news", tofon_model_name="masakhane/afrimt5_fr_fon_news" ):
        """
        Initialize the text-to-text model components.
        
        Args:
        model_name (str): Hugging Face model path.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tofrenchtokenizer = T5Tokenizer.from_pretrained(tofrench_model_name)
        self.tofrenchmodel = T5ForConditionalGeneration.from_pretrained(tofrench_model_name)
        self.tofontokenizer = T5Tokenizer.from_pretrained(tofrench_model_name)
        self.tofonmodel = T5ForConditionalGeneration.from_pretrained(tofon_model_name)
        self.tofrenchmodel.to(self.device)
        self.tofonmodel.to(self.device)

    def translate_to_french(self, text: str, max_length=512):
        """
        Translates Fon text to French using the loaded model.
        
        Args:
        text (str): Input text in Fon language.
        max_length (int): Maximum length of the output translation.
        
        Returns:
        str: translated text in French.
        """
        # Prepare the text input
        inputs = self.tofrenchtokenizer.encode("translate Fon to French: " + text, return_tensors="pt", max_length=max_length)
        inputs = inputs.to(self.device)

        # Generate translation using model
        outputs = self.tofrenchmodel.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        translated_text = self.tofrenchtokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

    def translate_to_fon(self, text: str, max_length=512):
        """
        Translates Fon text to French using the loaded model.
        
        Args:
        text (str): Input text in Fon language.
        max_length (int): Maximum length of the output translation.
        
        Returns:
        str: translated text in French.
        """
        # Prepare the text input
        inputs = self.tofontokenizer.encode("translate Fon to French: " + text, return_tensors="pt", max_length=max_length)
        inputs = inputs.to(self.device)

        # Generate translation using model
        outputs = self.tofonmodel.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        translated_text = self.tofontokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text
        


# # Example of using the FonTextToText class
# if __name__ == "__main__":
#     # Instantiate the translator with the specific Hugging Face model
#     model_name = "masakhane/afrimt5_fon_fr_news"
#     translator = FonTextToText(model_name=model_name)
    
#     # Example text in Fon
#     fon_text = "Write a sample Fon sentence here."
    
#     # Get the French translation
#     french_translation = translator.translate(fon_text)
#     print("Translated text:", french_translation)

# ### Explanation:
# 1. **Initialization of `FonTextToText`:**
#    - The `__init__` method initializes the tokenizer and model from the specified model name. It also sets the appropriate device (GPU or CPU) based on availability.

# 2. **Translation Method:**
#    - The `translate` function accepts Fon text and produces the French equivalent using the pre-trained T5 model from the Masakhane project. It preprocesses the text to fit the model's expected input format, generates predictions, and post-processes them into readable French text.

# 3. **Example Usage:**
#    - The `if __name__ == "__main__":` block provides an example of how to use this class. It initializes an instance with the Hugging Face model identifier and translates a sample Fon sentence.
     