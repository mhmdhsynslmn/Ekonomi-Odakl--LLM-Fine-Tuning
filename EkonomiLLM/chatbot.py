import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./economy-lora"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    load_in_8bit=True
)

model.eval()


print("💬 Economy Mini Chatbot")
print("Çıkmak için 'exit' yaz.\n")

while True:
    user_input = input("📰 Ekonomi Haberi / Soru: ")

    if user_input.lower() == "exit":
        print("👋 Çıkılıyor...")
        break

    prompt = f"""### Instruction:
Provide a concise economic analysis.

### Input:
{user_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=1.0
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    print("\n🤖 Model Yanıtı:")
    print(response)
    print("-" * 50)
