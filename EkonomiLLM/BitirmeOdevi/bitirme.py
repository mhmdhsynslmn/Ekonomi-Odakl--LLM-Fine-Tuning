import os
from pathlib import Path

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download("punkt")
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


#  HF Token

load_dotenv(Path(__file__).with_name("bitirme.env"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN veya HUGGINGFACE_HUB_TOKEN bulunamadı! Lütfen BitirmeOdevi/bitirme.env dosyasına ekleyin veya sistem ortam değişkeni olarak tanımlayın.")
login(token=HF_TOKEN)


#  MODEL VE TOKENIZER


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")


#  LoRA KONFİGÜRASYONU


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")



#  DATASET YÜKLEME

dataset_path = "economy_news.jsonl"
data = load_dataset("json", data_files=dataset_path)["train"]


# FORMATLAMA

def format_record(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\nTek cümlelik, resmi bir ekonomi haberi diliyle yanıt ver:\n"

    )
    return {"text": prompt}


def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

data = data.map(format_record)

tokenized_data = data.map(
    tokenize,
    batched=True,
    remove_columns=data.column_names
)



#  DATA COLLATOR


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# TRAINING ARGUMENTS


training_args = TrainingArguments(
    output_dir="./tinyllama-economy-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=500,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
    report_to="none"
)


# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator
)

# EĞİTİM BAŞLAT


trainer.train()

#  SKORU HESAPLAMA

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

model.eval()

bleu_score = compute__nltk(
    model=model,
    tokenizer=tokenizer,
    dataset=data,
    count=50
)

print("BLEU SCORE:", bleu_score)



#  MODELİ KAYDET

model.save_pretrained("./economy-lora")
tokenizer.save_pretrained("./economy-lora")
print("Model kaydedildi.")
