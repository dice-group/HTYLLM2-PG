# German monolingual model
from datasets import load_dataset

def load_german_dataset():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    return dataset