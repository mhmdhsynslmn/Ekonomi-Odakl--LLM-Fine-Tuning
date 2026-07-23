import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# =============================
# MODEL YÜKLE
# =============================
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "./economy-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
model = PeftModel.from_pretrained(base_model, lora_path).to(device)
model.eval()


# DATASET

dataset_path = "economy_news.jsonl"
data = load_dataset("json", data_files=dataset_path)["train"]


# BLEU FONKSİYONU

def compute__nltk(model, tokenizer, dataset, count=50):
    references = []
    hypotheses = []

    for sample in dataset.select(range(min(count, len(dataset)))):
        prompt = (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80)

        prediction = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        references.append([sample["output"].split()])
        hypotheses.append(prediction.split())

    smooth = SmoothingFunction().method7

    return corpus_bleu(
        references,
        hypotheses,
        weights=(0.5, 0.5),
        smoothing_function=smooth
    )


# BLEU HESAPLA

bleu_score = compute__nltk(model, tokenizer, data, count=50)
print("economy-lora BLEU SCORE:", bleu_score)
