# Ekonomi-Odaklı LLM İnce Ayar (Fine-Tuning)

Bu proje, ekonomi haberleri alanında özel bir dil modeli geliştirmek amacıyla hazırlanmıştır. TinyLlama tabanlı bir model üzerinde LoRA/peft ile ince ayar yapılmış ve değerlendirme adımları da projeye dahil edilmiştir.

## Amaç

- Ekonomi haberlerine yönelik metin üretimi yapmak
- Fine-tuning sürecini açık ve yeniden üretilebilir şekilde sunmak
- BLEU benzeri değerlendirme adımlarını projeye dahil etmek

## Proje Yapısı

- [EkonomiLLM/BitirmeOdevi/bitirme.py](EkonomiLLM/BitirmeOdevi/bitirme.py): Eğitim ve fine-tuning süreci
- [EkonomiLLM/chatbot.py](EkonomiLLM/chatbot.py): Model ile örnek konuşma/cevap üretme
- [EkonomiLLM/evaluate_bleu.py](EkonomiLLM/evaluate_bleu.py): Model değerlendirme betiği
- [EkonomiLLM/economy_news.jsonl](EkonomiLLM/economy_news.jsonl): Eğitim verisi

## Kurulum

1. Python 3.10+ sürümünü kurun.
2. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

3. Hugging Face token’ınızı yerelde tanımlayın:

Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

## Eğitim ve Çalıştırma

Eğitim sürecini başlatmak için:

```bash
python EkonomiLLM/BitirmeOdevi/bitirme.py
```

Modeli değerlendirmek için:

```bash
python EkonomiLLM/evaluate_bleu.py
```

## Güvenlik Notu

- Gizli anahtarlar ve token’lar asla repo içerisine eklenmemelidir.
- `.env` ve `bitirme.env` dosyaları `.gitignore` ile korunur.

## Not

Bu repo, büyük model ağırlıkları ve checkpoint dosyalarını içermez. İsterseniz ilerleyen adımlarda bunları da ekleyebilir ve daha kapsamlı bir üretim sürecine dönüştürebilirsiniz.
