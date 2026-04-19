from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    print("Loading trained model...")
    model_path = "./outputs/final_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    prompt = "Die Geschichte beginnt"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Text:")
    print(text)


if __name__ == "__main__":
    main()