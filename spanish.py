import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("\n Chat listo (Spanish supported)\n")

while True:
    prompt = input("Prompt: ")

    if prompt.lower() == "exit":
        break

    messages = [
        {
            "role": "system",
            "content": "Responde siempre en español de forma natural y conversacional."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    print("\nRespuesta:\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()
