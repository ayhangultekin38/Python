# =============================================================
# 📚 Multi-Source Literature Pipeline (with Semantic Scholar API Key)
# Author: Dr. Ayhan Gültekin
# Sources: Semantic Scholar + OpenAlex + CrossRef
# =============================================================

# -------------------------------------------------------------
# 1️⃣ GEREKLİ KÜTÜPHANELERxxxxxx
# -------------------------------------------------------------
try:
    import requests
except ImportError:
    import os
    os.system('pip install requests')
    import requests

import pandas as pd
import time
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 2️⃣ TEMEL AYARLAR
# -------------------------------------------------------------
query = "UAV path planning deep learning"
limit = 30  # her kaynaktan maksimum 30 sonuç
all_dataframes = []

# 🔑 S2 API Key (gizli tutun)
API_KEY = "QISpxKvu1CArkGUu7gDC75oTUHtdutv4PYeSHRx6"
HEADERS = {"x-api-key": API_KEY}

# -------------------------------------------------------------
# 3️⃣ SEMANTIC SCHOLAR
# -------------------------------------------------------------
print("🔹 Semantic Scholar'dan veri çekiliyor...")
sem_df = pd.DataFrame()
sem_url = "https://api.semanticscholar.org/graph/v1/paper/search"
sem_params = {
    "query": query,
    "limit": limit,
    "fields": "title,year,venue,authors,citationCount,doi,url"
}

for attempt in range(5):  # 429 için retry mekanizması
    try:
        response = requests.get(sem_url, params=sem_params, headers=HEADERS, timeout=25)
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", [])
            if data:
                sem_df = pd.DataFrame([{
                    "Source": "SemanticScholar",
                    "Title": p.get("title"),
                    "Year": p.get("year"),
                    "Venue": p.get("venue"),
                    "DOI": p.get("doi"),
                    "Citations": p.get("citationCount"),
                    "URL": p.get("url")
                } for p in data])
                print(f"✅ Semantic Scholar'dan {len(sem_df)} kayıt çekildi.")
            else:
                print("⚠️ Semantic Scholar yanıtı boş döndü.")
            break

        elif response.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"⏳ 429 Rate limit uyarısı — {wait} sn bekleniyor...")
            time.sleep(wait)

        else:
            print(f"⚠️ HTTP Hatası ({response.status_code}): {response.text}")
            break

    except Exception as e:
        print("❌ Semantic Scholar bağlantı hatası:", e)
        time.sleep(10)

all_dataframes.append(sem_df)

# -------------------------------------------------------------
# 4️⃣ OPENALEX
# -------------------------------------------------------------
print("🔹 OpenAlex'ten veri çekiliyor...")
oa_df = pd.DataFrame()
try:
    oa_url = "https://api.openalex.org/works"
    oa_params = {"search": query, "per-page": limit, "mailto": "example@example.com"}
    response = requests.get(oa_url, params=oa_params, timeout=20)
    if response.status_code == 200:
        result = response.json()
        data = result.get("results", [])
        if data:
            oa_df = pd.DataFrame([{
                "Source": "OpenAlex",
                "Title": p.get("title"),
                "Year": p.get("publication_year"),
                "Venue": p.get("host_venue", {}).get("display_name"),
                "DOI": p.get("doi"),
                "Citations": p.get("cited_by_count"),
                "URL": p.get("id")
            } for p in data])
            print(f"✅ OpenAlex'ten {len(oa_df)} kayıt çekildi.")
except Exception as e:
    print("❌ OpenAlex hata:", e)

all_dataframes.append(oa_df)

# -------------------------------------------------------------
# 5️⃣ CROSSREF
# -------------------------------------------------------------
print("🔹 CrossRef'ten veri çekiliyor...")
cr_df = pd.DataFrame()
try:
    cr_url = "https://api.crossref.org/works"
    cr_params = {"query": query, "rows": limit}
    response = requests.get(cr_url, params=cr_params, timeout=20)
    if response.status_code == 200:
        result = response.json()
        items = result.get("message", {}).get("items", [])
        if items:
            cr_df = pd.DataFrame([{
                "Source": "CrossRef",
                "Title": p.get("title", [""])[0],
                "Year": p.get("created", {}).get("date-parts", [[None]])[0][0],
                "Venue": p.get("container-title", [""])[0],
                "DOI": p.get("DOI"),
                "Citations": None,
                "URL": p.get("URL")
            } for p in items])
            print(f"✅ CrossRef'ten {len(cr_df)} kayıt çekildi.")
except Exception as e:
    print("❌ CrossRef hata:", e)

all_dataframes.append(cr_df)

# -------------------------------------------------------------
# 6️⃣ BİRLEŞTİRME & TEMİZLEME
# -------------------------------------------------------------
df = pd.concat(all_dataframes, ignore_index=True)
df.drop_duplicates(subset="DOI", inplace=True)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
print(f"\n📊 Toplam {len(df)} makale birleştirildi.\n")

# -------------------------------------------------------------
# 7️⃣ SCI-E DERGİ FİLTRESİ
# -------------------------------------------------------------
sci_publishers = ["Elsevier", "Springer", "IEEE", "Taylor", "Wiley", "MDPI", "SAGE"]
df["SCI_E"] = df["Venue"].apply(lambda v: any(pub.lower() in str(v).lower() for pub in sci_publishers))
sci_df = df[df["SCI_E"] == True]
print(f"📘 SCI-E benzeri dergilerde bulunan makale sayısı: {len(sci_df)}")

# -------------------------------------------------------------
# 8️⃣ CSV KAYITLARI
# -------------------------------------------------------------
df.to_csv("UAV_Literature_AllSources.csv", index=False)
sci_df.to_csv("UAV_Literature_SCI.csv", index=False)
print("💾 Veriler kaydedildi: UAV_Literature_AllSources.csv & UAV_Literature_SCI.csv")

# -------------------------------------------------------------
# 9️⃣ BASİT ANALİZ & GRAFİK
# -------------------------------------------------------------
if not df.empty:
    plt.figure(figsize=(10,4))
    df.groupby("Year").size().plot(kind="bar")
    plt.title("Yıllara Göre UAV + Deep Learning Yayın Dağılımı")
    plt.xlabel("Yıl")
    plt.ylabel("Makale Sayısı")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Görselleştirilecek veri bulunamadı.")
