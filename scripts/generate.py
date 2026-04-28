# python -m scripts.generate --lang de --prompt "Die Geschichte beginnt"
# python -m scripts.generate --lang fr --prompt "L'histoire commence"
# python -m scripts.generate --lang es --prompt "La historia comienza"
# python -m scripts.generate --lang it --prompt "La storia inizia"
# python -m scripts.generate --lang sv --prompt "Historien börjar"
# python -m scripts.generate --lang multi --prompt "Die Geschichte beginnt"/"L'histoire commence"/"La historia comienza"/"La storia inizia"/"Historien börjar"
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using trained model"
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language model to use (de, fr, es, etc.)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Die Geschichte beginnt",
        help="Input prompt text"
    )

    args = parser.parse_args()

    model_path = f"./outputs/{args.lang}_model"

    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(args.prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()