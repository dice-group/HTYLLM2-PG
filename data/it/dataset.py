# Italian monolingualdataset
from datasets import load_dataset

def load_italian_dataset():
    return load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")