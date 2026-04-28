# python -m scripts.generate --lang de --version v1 --prompt "Die Geschichte beginnt"
# python -m scripts.generate --lang fr --version v1 --prompt "L'histoire commence"
# python -m scripts.generate --lang es --version v1 --prompt "La historia comienza"
# python -m scripts.generate --lang it --version v1 --prompt "La storia inizia"
# python -m scripts.generate --lang sv --version v1 --prompt "Historien börjar"
# python -m scripts.generate --lang multi --version v1 --prompt "Die Geschichte beginnt"/"L'histoire commence"/"La historia comienza"/"La storia inizia"/"Historien börjar"
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
        help="Language model to use (de, fr, es, it, sv, multilingual)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Die Geschichte beginnt",
        help="Input prompt text"
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Model version (default: v1)"
    )

    args = parser.parse_args()

    base_path = "/data/HTYLLM2/models/distilgpt2"
    model_path = f"{base_path}/{args.version}/{args.lang}"

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