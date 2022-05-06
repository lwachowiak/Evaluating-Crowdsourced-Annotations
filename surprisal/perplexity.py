

# %%

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModelForCausalLM, AutoTokenizer

from math import exp

# %%

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

def perplexity(previous_annotations, new_annotation):

    # Construct prompt and completion string.
    prompt = "Please suggest some annotations for this image:\n\n- " + '\n- '.join(previous_annotations) + "\n- "
    completion = new_annotation
    text = prompt + completion

    # Compute length of prompt in tokens.
    prompt_length = tokenizer(prompt, return_tensors="pt")['input_ids'].size()[1]

    # Tokenize string.
    inputs = tokenizer(text, return_tensors="pt")

    # Duplicate input_ids to serve as labels.
    inputs['labels'] = inputs['input_ids'].clone().detach()

    # For all tokens in prompt, set label to -100 so they are excluded from loss calculation.
    for token_idx in range(prompt_length):
        inputs['labels'][0][token_idx] = -100

    # Compute loss.
    output = model(**inputs)
    return(exp(output.loss.item()))

# %%

previous_annotations = [
    "a dog",
    "a big brown dog",
    "a canine chasing a ball"
]
new_annotation = "labrador puppy"

perplexity(previous_annotations, new_annotation)
