# Ekonomi-Odaklı LLM İnce Ayar (Fine-Tuning)

Bu proje, ekonomi haberleri için ince ayar yapılmış bir LLM modeli geliştirmek amacıyla oluşturulmuştur.

## Proje İçeriği

- BitirmeOdevi/bitirme.py: fine-tuning eğitim süreci
- chatbot.py: model ile sohbet/cevap üretme örneği
- evaluate_bleu.py: BLEU skoru değerlendirme betiği
- economy_news.jsonl: eğitim verisi

## Kurulum

```bash
pip install -r requirements.txt
```

## Ortam Değişkeni

Hugging Face token’ınızı yerelde tanımlayın:

```bash
set HF_TOKEN=hf_your_token_here
```

## Kullanım

```bash
python EkonomiLLM/BitirmeOdevi/bitirme.py
```

## Not

- Gizli anahtarlar ve token’lar repo’ya eklenmemelidir.
- `.env` ve `bitirme.env` dosyaları `.gitignore` ile korunur.
