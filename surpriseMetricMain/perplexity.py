import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Perplexity:

    def __init__(self, previous_annotations) -> None:
        model_name = 'gpt2'
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = "Please suggest some annotations for this image:\n\n- " + '\n- '.join(
            previous_annotations) + "\n- "

    def score(self, new_annotation):
        completion = new_annotation
        text = self.prompt + completion

        # Compute length of prompt in tokens.
        prompt_length = self.tokenizer(self.prompt, return_tensors="pt")['input_ids'].size()[1]

        # Tokenize string.
        inputs = self.tokenizer(text, return_tensors="pt")

        # Duplicate input_ids to serve as labels.
        inputs['labels'] = inputs['input_ids'].clone().detach()

        # For all tokens in prompt, set label to -100 so they are excluded from loss calculation.
        for token_idx in range(prompt_length):
            inputs['labels'][0][token_idx] = -100

        # Compute loss.
        output = self.model(**inputs)
        return math.exp(output.loss.item())
