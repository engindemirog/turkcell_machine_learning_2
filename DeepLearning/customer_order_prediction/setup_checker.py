import subprocess
import sys

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sqlalchemy",
    "psycopg2-binary",
    "tensorflow",
    "scikit-learn",
    "imbalanced-learn",
    "fastapi",
    "uvicorn",
    "pydantic"
]

def check_and_install(package):
    try:
        __import__(package.replace("-", "_"))
        print(f"[✓] {package} yüklü.")
    except ImportError:
        print(f"[!] {package} bulunamadı. Yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print("🚀 Gerekli paketler kontrol ediliyor...")
    for pkg in REQUIRED_PACKAGES:
        check_and_install(pkg)
    print("✅ Tüm paketler hazır!")