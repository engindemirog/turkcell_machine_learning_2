# Müşteri Kategori Bazlı Satın Alma Tahmin Modeli

Bu proje, Northwind veritabanındaki müşterilerin geçmiş satın alma davranışlarını analiz ederek, yeni ürün kategorilerinde satın alma potansiyellerini tahmin eden bir derin öğrenme modeli içerir.

## Proje Yapısı

```
.
├── config/
│   └── config.py           # Konfigürasyon ayarları
├── data/
│   ├── raw/               # Ham veriler
│   └── processed/         # İşlenmiş veriler
├── models/
│   └── model.py          # Model tanımlamaları
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py   # Veritabanı işlemleri
│   │   └── preprocessing.py  # Veri ön işleme
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Özellik mühendisliği
│   └── training/
│       ├── __init__.py
│       └── train.py      # Model eğitimi
├── notebooks/
│   └── analysis.ipynb    # Veri analizi ve görselleştirme
├── requirements.txt      # Proje bağımlılıkları
└── main.py              # Ana uygulama
```

## Kurulum

1. Python 3.12.4 sürümünü yükleyin
2. Sanal ortam oluşturun ve aktifleştirin:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

1. `.env` dosyasını oluşturun ve veritabanı bağlantı bilgilerinizi ekleyin:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=northwind
   DB_USER=your_username
   DB_PASSWORD=your_password
   ```

2. Ana uygulamayı çalıştırın:
   ```bash
   python main.py
   ```

## Model Açıklaması

Bu proje, müşterilerin geçmiş satın alma davranışlarını analiz ederek, yeni ürün kategorilerinde satın alma potansiyellerini tahmin eden bir yapay sinir ağı modeli kullanır. Model, aşağıdaki özellikleri kullanır:

- Müşterinin toplam harcama miktarı
- Kategori bazlı harcama dağılımı
- Son satın alma tarihi
- Satın alma sıklığı
- Ortalama sipariş büyüklüğü 