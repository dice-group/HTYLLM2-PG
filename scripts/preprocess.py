# Main preprocessing pipeline
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM


# def main():
#     print("Loading dataset...")
#     dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

#     print("Sample data:")
#     sample = dataset["train"][10]["text"]
#     print(sample)

#     print("\nLoading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")

#     print("Tokenizing sample...")
#     tokens = tokenizer(sample)
#     print(tokens)

#     print("\nLoading model...")
#     model = AutoModelForCausalLM.from_pretrained("gpt2")

#     print("Model loaded successfully!")


# if __name__ == "__main__":
#     main()

from data.de.dataset import load_german_dataset
from utils.preprocessing import prepare_dataset


def main():
    dataset = load_german_dataset()
    tokenized_dataset, _ = prepare_dataset(dataset)

    print(tokenized_dataset[0])


if __name__ == "__main__":
    main()