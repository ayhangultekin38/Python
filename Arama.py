# =============================================================
# 📚 Multi-Source Literature Pipeline
# Author: Dr. Ayhan Gültekin
# Paper: A Time-Aware Bipartite Graph and ML Framework for
#        Analyzing Student–Employer Interactions in a Career Fair
# Sources: Semantic Scholar + OpenAlex + CrossRef
# =============================================================
# 💡 ÇALIŞMA MANTIĞI:
#    - İlk çalıştırma : API'den çeker → CareerFair_Literature_AllSources.csv'ye kaydeder
#    - Sonraki çalıştırmalar : CSV'den okur, API'ye gitmez
#    - Yeniden çekmek istersen : FORCE_FETCH = True yap
# =============================================================

# -------------------------------------------------------------
# 1️⃣ GEREKLİ KÜTÜPHANELER
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
from dotenv import load_dotenv
import os

# -------------------------------------------------------------
# 2️⃣ TEMEL AYARLAR
# -------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("SS_API_KEY")
HEADERS = {"X-API-KEY": API_KEY} if API_KEY else {}
LIMIT = 30

CACHE_FILE = "CareerFair_Literature_AllSources.csv"
FORCE_FETCH = False  # True yapınca API'den yeniden çeker, False = cache kullan

QUERIES = {
    "S1_CareerFair":      "career fair student employer interaction recruitment",
    "S2_StudentMatching": "student employer matching recommendation system job fair",
    "S3_ML_CareerEdu":    "machine learning educational analytics career prediction student",
    "S4_BipartiteGraph":  "bipartite graph temporal interaction network recommendation",
    "S4_DynamicGraph":    "dynamic graph neural network time-aware link prediction",
}

# -------------------------------------------------------------
# 3️⃣ FONKSİYON: SEMANTIC SCHOLAR
# -------------------------------------------------------------
def fetch_semantic_scholar(query, label, limit=LIMIT):
    print(f"  🔹 [S2] {label}")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,publicationVenue,authors,citationCount,externalIds,url"
    }
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    df = pd.DataFrame([{
                        "Source":    "SemanticScholar",
                        "Section":   label,
                        "Title":     p.get("title"),
                        "Year":      p.get("year"),
                        "Venue":     p.get("publicationVenue", {}).get("name") if p.get("publicationVenue") else None,
                        "DOI":       p.get("externalIds", {}).get("DOI") if p.get("externalIds") else None,
                        "Citations": p.get("citationCount"),
                        "URL":       p.get("url")
                    } for p in data])
                    print(f"     ✅ {len(df)} kayıt")
                    return df
                print("     ⚠️ Boş yanıt")
                return pd.DataFrame()
            elif r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"     ⏳ Rate limit — {wait}sn bekleniyor...")
                time.sleep(wait)
            else:
                print(f"     ⚠️ HTTP {r.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"     ❌ {e}")
            time.sleep(10)
    return pd.DataFrame()

# -------------------------------------------------------------
# 4️⃣ FONKSİYON: OPENALEX
# -------------------------------------------------------------
def fetch_openalex(query, label, limit=LIMIT):
    print(f"  🔹 [OA] {label}")
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": limit, "mailto": "example@example.com"},
            timeout=20
        )
        if r.status_code == 200:
            data = r.json().get("results", [])
            if data:
                df = pd.DataFrame([{
                    "Source":    "OpenAlex",
                    "Section":   label,
                    "Title":     p.get("title"),
                    "Year":      p.get("publication_year"),
                    "Venue":     ((p.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "DOI":       p.get("doi"),
                    "Citations": p.get("cited_by_count"),
                    "URL":       p.get("id")
                } for p in data])
                print(f"     ✅ {len(df)} kayıt")
                return df
    except Exception as e:
        print(f"     ❌ {e}")
    return pd.DataFrame()

# -------------------------------------------------------------
# 5️⃣ FONKSİYON: CROSSREF
# -------------------------------------------------------------
def fetch_crossref(query, label, limit=LIMIT):
    print(f"  🔹 [CR] {label}")
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit},
            timeout=20
        )
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                df = pd.DataFrame([{
                    "Source":    "CrossRef",
                    "Section":   label,
                    "Title":     p.get("title", [""])[0],
                    "Year":      p.get("created", {}).get("date-parts", [[None]])[0][0],
                    "Venue":     p.get("container-title", [""])[0],
                    "DOI":       p.get("DOI"),
                    "Citations": None,
                    "URL":       p.get("URL")
                } for p in items])
                print(f"     ✅ {len(df)} kayıt")
                return df
    except Exception as e:
        print(f"     ❌ {e}")
    return pd.DataFrame()

# -------------------------------------------------------------
# 6️⃣ VERİ ÇEKİMİ VEYA CACHE'DEN OKU
# -------------------------------------------------------------
if os.path.exists(CACHE_FILE) and not FORCE_FETCH:
    print(f"📂 Cache bulundu → '{CACHE_FILE}' okunuyor (API'ye gidilmiyor)")
    df = pd.read_csv(CACHE_FILE)
    print(f"   ✅ {len(df)} makale yüklendi.")
else:
    print("🌐 API'den veri çekiliyor...\n")
    all_dataframes = []
    for label, query in QUERIES.items():
        print(f"\n📌 Sorgu: {label}")
        all_dataframes.append(fetch_semantic_scholar(query, label))
        all_dataframes.append(fetch_openalex(query, label))
        all_dataframes.append(fetch_crossref(query, label))
        time.sleep(2)

    df = pd.concat(all_dataframes, ignore_index=True)
    df.drop_duplicates(subset="DOI", inplace=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"] >= 2010]
    df = df.dropna(subset=["Title"])
    df.to_csv(CACHE_FILE, index=False)
    print(f"\n💾 {len(df)} makale '{CACHE_FILE}' dosyasına kaydedildi.")

print(f"\n📊 Toplam {len(df)} makale işleniyor.\n")

# -------------------------------------------------------------
# 7️⃣ SCI-E DERGİ FİLTRESİ
# -------------------------------------------------------------
sci_publishers = ["Elsevier", "Springer", "IEEE", "Taylor", "Wiley", "MDPI", "SAGE", "ACM", "Nature", "Oxford"]
df["SCI_E"] = df["Venue"].apply(
    lambda v: any(pub.lower() in str(v).lower() for pub in sci_publishers)
)
sci_df = df[df["SCI_E"] == True]
sci_df.to_csv("CareerFair_Literature_SCI.csv", index=False)
print(f"📘 SCI-E benzeri: {len(sci_df)} makale → CareerFair_Literature_SCI.csv")

# Bölüm bazlı özet
summary = df.groupby("Section")["Title"].count().reset_index()
summary.columns = ["Bölüm", "Makale Sayısı"]
print("\n📋 Bölüm bazlı dağılım:")
print(summary.to_string(index=False))

# -------------------------------------------------------------
# 8️⃣ GRAFİKLER
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.groupby("Year").size().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Yıllara Göre Yayın Dağılımı")
axes[0].set_xlabel("Yıl")
axes[0].set_ylabel("Makale Sayısı")

df.groupby("Section").size().plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Related Work Bölümlerine Göre Dağılım")
axes[1].set_xlabel("Makale Sayısı")

plt.tight_layout()
plt.savefig("CareerFair_Literature_Graph.png", dpi=150)
plt.show()
print("📊 Grafik kaydedildi: CareerFair_Literature_Graph.png")

# -------------------------------------------------------------
# 9️⃣ CONNECTED PAPERS İÇİN MERKEZ MAKALELER
# -------------------------------------------------------------
print("\n" + "="*60)
print("🔗 CONNECTED PAPERS İÇİN ÖNERİLEN MERKEZ MAKALELER")
print("="*60)

cp_rows = []
for section in df["Section"].unique():
    section_df = df[df["Section"] == section].copy()
    section_df = section_df.dropna(subset=["Citations", "DOI"])
    section_df = section_df.sort_values("Citations", ascending=False)
    top2 = section_df.head(2)

    print(f"\n📂 {section}")
    for _, row in top2.iterrows():
        doi = str(row["DOI"]).replace("https://doi.org/", "")
        cp_url = f"https://www.connectedpapers.com/main/{doi}"
        print(f"  📄 {str(row['Title'])[:80]}...")
        print(f"     Yıl: {int(row['Year'])} | Atıf: {int(row['Citations'])}")
        print(f"     🔗 {cp_url}")
        cp_rows.append({
            "Section":             section,
            "Title":               row["Title"],
            "Year":                row["Year"],
            "Citations":           row["Citations"],
            "DOI":                 doi,
            "ConnectedPapers_URL": cp_url
        })

cp_df = pd.DataFrame(cp_rows)
cp_df.to_csv("ConnectedPapers_SeedList.csv", index=False)
print("\n💾 Kaydedildi: ConnectedPapers_SeedList.csv")
print("👆 Linklere tıklayarak Connected Papers'ı açabilirsiniz.")
