# Ekonomi-Odaklı LLM İnce Ayar (Fine-Tuning)

Bu proje, ekonomi haberleri alanında özelleşmiş bir dil modeli geliştirmek amacıyla hazırlanmıştır. TinyLlama tabanlı model üzerinde **LoRA / PEFT** (Parameter-Efficient Fine-Tuning) yöntemleriyle ince ayar gerçekleştirilmiş ve kapsamlı değerlendirme adımları projeye dahil edilmiştir.

## Proje Amaçları

- Ekonomi haberleri bağlamında özelleşmiş metin üretimi gerçekleştirmek.
- Fine-tuning sürecini şeffaf, anlaşılır ve yeniden üretilebilir bir şekilde sunmak.
- BLEU metrikleri ile model performansını değerlendirmek.

## Proje Yapısı

```
EkonomiLLM/
├── BitirmeOdevi/
│   └── bitirme.py            # Model eğitimi ve fine-tuning süreci
├── chatbot.py                # Model ile örnek sohbet ve cevap üretme betiği
├── evaluate_bleu.py          # Model değerlendirme ve BLEU skoru hesaplama
└── economy_news.jsonl        # Eğitim için veri seti
```

## Kurulum ve Başlangıç

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. **Python Gereksinimi:**
Sisteminizde Python 3.10 veya üzeri bir sürümün kurulu olduğundan emin olun.
2. **Bağımlılıkları Yükleyin:**

```bash
pip install -r requirements.txt
```

3. **Hugging Face Token Tanımlama:**
Model indirme ve doğrulama işlemleri için Hugging Face token'ınızı terminal ortamında tanımlayın:

- **Windows PowerShell için:**

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

- **Linux / macOS için:**

```bash
export HF_TOKEN="hf_your_token_here"
```

## Eğitim ve Çalıştırma

- **Eğitim Sürecini Başlatmak İçin:**

```bash
python EkonomiLLM/BitirmeOdevi/bitirme.py
```

- **Modeli Değerlendirmek (BLEU) İçin:**

```bash
python EkonomiLLM/evaluate_bleu.py
```

## Not

> **Depo Boyutu Hakkında:** Bu depo, büyük model ağırlıklarını ve checkpoint dosyalarını içermez. İhtiyaç duymanız halinde ilerleyen adımlarda ağırlıkları ekleyebilir veya projeyi uçtan uca bir üretim sürecine dönüştürebilirsiniz.
