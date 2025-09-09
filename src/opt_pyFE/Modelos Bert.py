#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scraper de titulares por ticker + Análisis de Sentimiento con BERT (FinancialBERT)
-----------------------------------------------------------------------------------
1) Configuras TICKERS y fechas.
2) Scraping (Google News, Bing News, Yahoo Finance RSS).
3) Sentimiento con BERT.
4) Serie de tiempo por ticker/fecha.
5) Gráficas en pantalla (no se guardan archivos). Sin forzar backend externo.

Dependencias:
  pip install aiohttp feedparser yfinance rapidfuzz python-dateutil pandas tldextract \
              tenacity nest_asyncio transformers matplotlib
"""

# ==============================
# Silenciar logs de TensorFlow (si Transformers cae en TF)
# ==============================
import os as os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")  # fuerza a no usar TF

# ==========================================
# Compatibilidad con Spyder/Jupyter/Windows
# ==========================================
import sys
import asyncio
try:
    import nest_asyncio  # pip install nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ==============================
# IMPORTS GENERALES
# ==============================
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import aiohttp
import feedparser
import pandas as pd
import tldextract
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rapidfuzz import fuzz

# --- Matplotlib (SIN forzar backends de ventana; intentar inline si hay IPython) ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    # Si estamos en IPython/Jupyter/Spyder, fuerza inline (no ventanas externas)
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    from IPython import get_ipython  # type: ignore
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("png")
except Exception:
    # Si no hay IPython, simplemente usará el backend por defecto del sistema
    pass

# ==============================
# CONFIGURACIÓN PRINCIPAL
# ==============================

# <<< EDITA AQUÍ: Lista de tickers de entrada >>>
TICKERS: List[str] = [
    "LIVEPOLC-1.MX",  # El Puerto de Liverpool (BMV)
    "AMZN",           # Amazon
    "AAPL",           # Apple
    "NVDA",           # Nvidia
    "GOOGL",          # Alphabet
]

# --- Filtro por FECHA de publicación (ISO "YYYY-MM-DD"); None desactiva ---
DATE_FROM: Optional[str] = "2025-01-01"
DATE_TO:   Optional[str] = "2025-09-09"

# Para obtener MÁS noticias, conserva items sin fecha publicada
INCLUDE_IF_NO_PUBLISHED = True

# --- Modelo de sentimiento (Hugging Face) ---
HF_MODEL_NAME = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
# Alternativa: # HF_MODEL_NAME = "ProsusAI/finbert"

DEVICE = "-1"        # CPU-only
BATCH_SIZE = 32      # Lote de inferencia

# Umbral opcional de neutralidad (None = desactivado)
NEUTRALITY_THRESHOLD: Optional[float] = None
# Ejemplo: NEUTRALITY_THRESHOLD = 0.55

# --- Scraper (idioma/país forzado a EN-US) ---
GOOGLE_NEWS_HL = "en-US"
GOOGLE_NEWS_GL = "US"
GOOGLE_NEWS_CEID = "US:en"
BING_NEWS_SETLANG = "en-us"
TRY_YAHOO_RSS = True

# --- Control de scraping (más cobertura) ---
RATE_LIMIT_DELAY = 0.4
MAX_NEWS_PER_TICKER = 1000
DEDUP_SIM_THRESHOLD = 82

# Filtro estricto por aparición del nombre/ticker en el TÍTULO
# Para traer MÁS resultados, lo dejamos en False (confiamos en la query del feed).
STRICT_TITLE_FILTER = False

# ==============================
# UTILIDADES SCRAPER
# ==============================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def pick_user_agent(i: int) -> str:
    return USER_AGENTS[i % len(USER_AGENTS)]

def domain_from_url(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        root = ".".join(part for part in [ext.domain, ext.suffix] if part)
        return root or urlparse(url).netloc
    except Exception:
        return urlparse(url).netloc or ""

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()

def parse_date_param(d: Optional[str], end: bool = False) -> Optional[pd.Timestamp]:
    """
    Convierte 'YYYY-MM-DD' a Timestamp UTC.
    Si end=True, lo mueve al final del día.
    """
    if not d:
        return None
    try:
        ts = pd.to_datetime(d, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        if end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return ts
    except Exception:
        return None

DATE_FROM_TS = parse_date_param(DATE_FROM, end=False)
DATE_TO_TS   = parse_date_param(DATE_TO,   end=True)

def in_date_range(pub_iso: Optional[str]) -> bool:
    """
    Valida si 'published' (ISO) cae dentro del rango [DATE_FROM_TS, DATE_TO_TS].
    Si no hay filtros activos, True. Si no hay fecha y hay filtros -> INCLUDE_IF_NO_PUBLISHED.
    """
    filters_active = (DATE_FROM_TS is not None) or (DATE_TO_TS is not None)
    if not filters_active:
        return True
    if not pub_iso:
        return INCLUDE_IF_NO_PUBLISHED
    try:
        pt = pd.to_datetime(pub_iso, utc=True, errors="coerce")
        if pd.isna(pt):
            return INCLUDE_IF_NO_PUBLISHED
    except Exception:
        return INCLUDE_IF_NO_PUBLISHED
    if DATE_FROM_TS is not None and pt < DATE_FROM_TS:
        return False
    if DATE_TO_TS is not None and pt > DATE_TO_TS:
        return False
    return True

def dedup_titles(rows: List[Dict], sim_threshold: int = DEDUP_SIM_THRESHOLD) -> List[Dict]:
    kept, seen_titles = [], []
    for r in rows:
        title = normalize_text(r.get("title", ""))
        if not title:
            continue
        is_dup = any(fuzz.token_set_ratio(title, st) >= sim_threshold for st in seen_titles)
        if not is_dup:
            kept.append(r)
            seen_titles.append(title)
    return kept

def parse_date_fuzzy(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.isoformat()
    except Exception:
        return None

def strip_exchange_suffix(t: str) -> str:
    return re.sub(r"\.[A-Z]{1,4}$", "", (t or "").strip())

# ==============================
# YFINANCE (OBTENER NOMBRE)
# ==============================
def resolve_company_name(ticker: str) -> Tuple[str, List[str]]:
    """
    Devuelve (company_name, synonyms_for_query).
    Fallback: base del ticker + variantes limpias.
    """
    base = strip_exchange_suffix(ticker)
    try:
        tkr = yf.Ticker(ticker)
        info = {}
        try:
            info = tkr.get_info()
        except Exception:
            try:
                info = tkr.info
            except Exception:
                info = {}
        name_candidates = []
        if isinstance(info, dict):
            for key in ("longName", "shortName", "name"):
                v = normalize_text(info.get(key, ""))
                if v:
                    name_candidates.append(v)
        seen, dedup = set(), []
        for x in name_candidates:
            if x not in seen:
                dedup.append(x); seen.add(x)
        company = dedup[0] if dedup else base

        synonyms = set([company])
        s_clean = re.sub(r"S\.?A\.?B?\.?\s*de\s*C\.?V\.?", "", company, flags=re.IGNORECASE).strip()
        s_clean = re.sub(r"S\.?A\.?\s*de\s*C\.?V\.?", "", s_clean, flags=re.IGNORECASE).strip()
        s_clean = re.sub(r"Sociedad Anónima.*", "", s_clean, flags=re.IGNORECASE).strip()
        if s_clean and s_clean.lower() != company.lower():
            synonyms.add(s_clean)
        synonyms.add(base)
        for kw in ["stock", "shares", "earnings", "results", "guidance",
                   "outlook", "rating", "upgrade", "downgrade"]:
            synonyms.add(f"{company} {kw}")
        return company, sorted(synonyms)
    except Exception:
        return base, [base]

# ==============================
# GENERACIÓN DE FUENTES (RSS)
# ==============================
def google_news_rss_queries(query: str) -> List[str]:
    """
    Variantes para ampliar cobertura (normal + when:365d).
    """
    base_q = quote_plus(query)
    q_when = quote_plus(f"({query}) when:365d")
    return [
        f"https://news.google.com/rss/search?q={base_q}"
        f"&hl={GOOGLE_NEWS_HL}&gl={GOOGLE_NEWS_GL}&ceid={GOOGLE_NEWS_CEID}",
        f"https://news.google.com/rss/search?q={q_when}"
        f"&hl={GOOGLE_NEWS_HL}&gl={GOOGLE_NEWS_GL}&ceid={GOOGLE_NEWS_CEID}",
    ]

def bing_news_rss_queries(query: str) -> List[str]:
    q = quote_plus(query)
    return [f"https://www.bing.com/news/search?q={q}&format=RSS&setlang={BING_NEWS_SETLANG}"]

def yahoo_finance_rss(symbol: str) -> List[str]:
    """
    Varias ventanas de edad para ampliar resultados.
    """
    s = quote_plus(symbol)
    return [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={s}&region=US&lang=en-US",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={s}&region=US&lang=en-US&age=1d",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={s}&region=US&lang=en-US&age=7d",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={s}&region=US&lang=en-US&age=30d",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={s}&region=US&lang=en-US&age=90d",
    ]

def build_query_strings(ticker: str, synonyms: List[str]) -> List[str]:
    """
    Genera varias queries para repartir la búsqueda.
    """
    base = strip_exchange_suffix(ticker)
    parts = [base] + [s.strip() for s in synonyms if s and s.strip()]
    uniq, seen = [], set()
    for p in parts:
        lp = p.lower()
        if lp not in seen:
            uniq.append(p); seen.add(lp)
    queries = []
    # 1) Sólo ticker
    queries.append(base)
    # 2) Nombres compuestos exactos
    for p in uniq:
        if " " in p:
            queries.append(f"\"{p}\"")
    # 3) Paquetes OR con tokens simples
    chunk = []
    for p in uniq:
        if " " not in p:
            chunk.append(p)
        if len(chunk) >= 6:
            queries.append(" OR ".join(chunk)); chunk = []
    if chunk:
        queries.append(" OR ".join(chunk))
    return queries

# ==============================
# DESCARGA Y PARSING (ASYNC)
# ==============================
class FetchError(Exception):
    pass

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, FetchError))
)
async def fetch_text(session: aiohttp.ClientSession, url: str, ua_index: int) -> str:
    from aiohttp import ClientTimeout
    headers = {"User-Agent": pick_user_agent(ua_index)}
    await asyncio.sleep(RATE_LIMIT_DELAY)
    async with session.get(url, headers=headers, timeout=ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            raise FetchError(f"HTTP {resp.status} for {url}")
        return await resp.text()

def parse_feed(content: str) -> List[Dict]:
    parsed = feedparser.parse(content)
    items = []
    for e in parsed.entries:
        title = normalize_text(getattr(e, "title", ""))
        link = normalize_text(getattr(e, "link", ""))
        published = None
        for k in ("published", "updated", "pubDate", "dc_date"):
            v = getattr(e, k, None)
            if v:
                published = parse_date_fuzzy(v)
                if published:
                    break
        if title and link:
            items.append({"title": title, "link": link, "published": published})
    return items

async def scrape_for_ticker(session: aiohttp.ClientSession, ticker: str,
                            company: str, synonyms: List[str], ua_index: int) -> List[Dict]:
    all_rows: List[Dict] = []

    # Construir queries repartidas
    query_blocks = build_query_strings(ticker, synonyms)

    g_urls, b_urls, y_urls = [], [], []
    for q in query_blocks:
        g_urls.extend(google_news_rss_queries(q))
        b_urls.extend(bing_news_rss_queries(q))
    if TRY_YAHOO_RSS:
        y_urls = yahoo_finance_rss(ticker)

    rss_urls = g_urls + b_urls + y_urls

    for idx, url in enumerate(rss_urls):
        try:
            text = await fetch_text(session, url, ua_index + idx)
            for it in parse_feed(text):
                all_rows.append({
                    "ticker": ticker,
                    "company": company,
                    "source": domain_from_url(it["link"]),
                    "title": it["title"],
                    "link": it["link"],
                    "published": it.get("published"),
                    "fetched_at": now_iso(),
                })
        except Exception:
            continue

    if STRICT_TITLE_FILTER:
        base = strip_exchange_suffix(ticker).lower()
        name_tokens = {s.lower() for s in synonyms if s}
        filtered = []
        for r in all_rows:
            t = r["title"].lower()
            keep = base in t or any((len(nm) >= 3 and nm in t) for nm in name_tokens)
            if keep:
                filtered.append(r)
    else:
        filtered = all_rows  # confiar en la query del feed para mayor recall

    # Filtro de fechas y deduplicación
    filtered = [r for r in filtered if in_date_range(r.get("published"))]
    filtered = dedup_titles(filtered, sim_threshold=DEDUP_SIM_THRESHOLD)

    # Ordenar por fecha (publicada o, si no hay, por fetched_at)
    def sort_key(x):
        return x.get("published") or x.get("fetched_at")
    filtered.sort(key=sort_key, reverse=True)

    return filtered[:MAX_NEWS_PER_TICKER]

# ==============================
# PIPELINE SCRAPER
# ==============================
@dataclass
class TickerResult:
    ticker: str
    company: str
    rows: List[Dict]

async def process_tickers(tickers: List[str]) -> List[TickerResult]:
    resolved: Dict[str, Tuple[str, List[str]]] = {}
    for t in tickers:
        company, synonyms = resolve_company_name(t)
        resolved[t] = (company, synonyms)

    connector = aiohttp.TCPConnector(limit=16, ssl=False)
    timeout = aiohttp.ClientTimeout(total=40)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, t in enumerate(tickers):
            company, synonyms = resolved[t]
            tasks.append(scrape_for_ticker(session, t, company, synonyms, ua_index=i))
        results: List[List[Dict]] = await asyncio.gather(*tasks, return_exceptions=True)

    final: List[TickerResult] = []
    for t, res in zip(tickers, results):
        comp, _ = resolved[t]
        if isinstance(res, Exception):
            final.append(TickerResult(ticker=t, company=comp, rows=[]))
        else:
            final.append(TickerResult(ticker=t, company=comp, rows=res))
    return final

def to_dataframe(results: List[TickerResult]) -> pd.DataFrame:
    all_rows = []
    for r in results:
        all_rows.extend(r.rows)
    cols = ["ticker", "company", "source", "title", "link", "published", "fetched_at"]
    df = pd.DataFrame(all_rows, columns=cols)
    if not df.empty:
        # Fecha efectiva para análisis (publicada si existe; si no, fetched_at)
        eff = df["published"].fillna(df["fetched_at"])
        dt = pd.to_datetime(eff, utc=True, errors="coerce")
        df["effective_datetime_utc"] = dt
        df["date_utc"] = dt.dt.date
    return df

# (DESACTIVADO) Exportación a CSV/Parquet
# def export_raw(df: pd.DataFrame, prefix: str) -> Tuple[str, str]:
#     ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
#     csv_path = os.path.join("news_output", f"{prefix}_{ts}.csv")
#     parquet_path = os.path.join("news_output", f"{prefix}_{ts}.parquet")
#     df.to_csv(csv_path, index=False, encoding="utf-8")
#     df.to_parquet(parquet_path, index=False)
#     return csv_path, parquet_path

# ==============================
# SENTIMIENTO (Transformers, CPU)
# ==============================
def build_sentiment_pipeline(model_name: str = HF_MODEL_NAME,
                             device: str = DEVICE,
                             batch_size: int = BATCH_SIZE):
    """
    Devuelve (pipeline, effective_device, batch_size, framework_str).
    """
    from transformers import pipeline
    effective_device = -1  # CPU
    framework = None
    try:
        import torch  # noqa: F401
        framework = "pt"
    except Exception:
        framework = None

    pl = pipeline(
        task="text-classification",
        model=model_name,
        device=effective_device,   # -1 (CPU)
        truncation=True,
        padding=True,
        framework=framework
    )
    print(f"Modelo cargado en CPU. Framework: {framework or 'auto'}")
    return pl, effective_device, batch_size, (framework or "auto")

def scores_from_pipeline_output(item_list: List[Dict],
                                neutrality_threshold: Optional[float] = NEUTRALITY_THRESHOLD
                                ) -> Tuple[float, float, float, str, str]:
    """
    Devuelve: (pos, neg, neu, label_tri, label_forced_pn)
    """
    pos = neg = neu = 0.0
    for d in item_list:
        lab = str(d.get("label", "")).lower()
        sc = float(d.get("score", 0.0))
        if "positive" in lab or lab == "pos":
            pos = sc
        elif "negative" in lab or lab == "neg":
            neg = sc
        elif "neutral"  in lab or lab == "neu":
            neu = sc

    tri = {"positive": pos, "neutral": neu, "negative": neg}
    label_argmax = max(tri, key=tri.get)
    best = tri[label_argmax]

    if neutrality_threshold is not None and best < float(neutrality_threshold):
        label_argmax = "neutral"

    forced_pn = "positive" if pos >= neg else "negative"
    return pos, neg, neu, label_argmax, forced_pn

def run_sentiment(df: pd.DataFrame,
                  text_col: str = "title",
                  model_name: str = HF_MODEL_NAME,
                  device: str = DEVICE,
                  batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """
    Anota df con columnas de scores y etiquetas.
    """
    if df.empty:
        return df.assign(score_positive=[], score_negative=[], score_neutral=[],
                         label_tri=[], label_forced_pn=[])
    pl, eff_dev, bs, fw = build_sentiment_pipeline(model_name, device, batch_size)
    texts = df[text_col].astype(str).tolist()

    all_outputs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        out = pl(batch, top_k=None)  # todas las etiquetas+scores
        all_outputs.extend(out)

    pos_list, neg_list, neu_list, tri_list, forced_list = [], [], [], [], []
    for o in all_outputs:
        pos, neg, neu, label_tri, forced_pn = scores_from_pipeline_output(o, NEUTRALITY_THRESHOLD)
        pos_list.append(pos); neg_list.append(neg); neu_list.append(neu)
        tri_list.append(label_tri); forced_list.append(forced_pn)

    df = df.copy()
    df["score_positive"]   = pos_list
    df["score_negative"]   = neg_list
    df["score_neutral"]    = neu_list
    df["label_tri"]        = tri_list
    df["label_forced_pn"]  = forced_list
    return df

# ==============================
# AGREGACIÓN / SERIES DE TIEMPO
# ==============================
def build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serie por ticker-fecha con promedios y proporciones.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker","date_utc","n_headlines","mean_pos","mean_neg","mean_neu",
            "net_sentiment","share_pos","share_neu","share_neg","share_pos_forced"
        ])

    gcols = ["ticker", "date_utc"]
    tmp = df.copy()
    tmp["is_pos"] = (tmp["label_tri"] == "positive").astype(int)
    tmp["is_neu"] = (tmp["label_tri"] == "neutral").astype(int)
    tmp["is_neg"] = (tmp["label_tri"] == "negative").astype(int)
    tmp["is_pos_forced"] = (tmp["label_forced_pn"] == "positive").astype(int)

    agg = tmp.groupby(gcols, as_index=False).agg(
        n_headlines=("title", "count"),
        mean_pos=("score_positive", "mean"),
        mean_neg=("score_negative", "mean"),
        mean_neu=("score_neutral", "mean"),
        share_pos=("is_pos", "mean"),
        share_neu=("is_neu", "mean"),
        share_neg=("is_neg", "mean"),
        share_pos_forced=("is_pos_forced", "mean"),
    )
    agg["net_sentiment"] = agg["mean_pos"] - agg["mean_neg"]
    agg = agg.sort_values(["ticker", "date_utc"])
    return agg

# ==============================
# 7) GRÁFICAS DE BARRAS (colores + eje fechas sin solapamiento)
# ==============================
def _colors_from_values(values):
    return ["green" if v > 0 else "red" if v < 0 else "gray" for v in values]

def _setup_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.margins(x=0.01)

# --- helper: promedio ponderado por titulares (sin groupby.apply; evita FutureWarning) ---
def _portfolio_weighted_df(ts_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = ts_df[["date_utc", metric, "n_headlines"]].copy()
    df["w_val"] = df[metric] * df["n_headlines"]
    agg = (
        df.groupby("date_utc", as_index=False)
          .agg(w_sum=("w_val", "sum"), w=("n_headlines", "sum"))
    )
    agg[metric] = agg["w_sum"] / agg["w"]
    return agg[["date_utc", metric]]

def plot_sentiment_bars(ts_df: pd.DataFrame, metric: str = "net_sentiment"):
    """
    Muestra:
      - Barras por día para cada ticker (colores por signo del sentimiento).
      - Portafolio igual-ponderado.
      - Portafolio ponderado por cantidad de titulares.
    Eje X compacto con fechas para evitar solapamiento.
    """
    if ts_df.empty:
        print("No hay datos en ts para graficar.")
        return

    ts_df = ts_df.sort_values(["ticker", "date_utc"])

    # ---------- 7.1 Gráfica por ticker ----------
    for tkr, sub in ts_df.groupby("ticker"):
        sub = sub.sort_values("date_utc")
        x_dt = pd.to_datetime(sub["date_utc"])
        y = sub[metric].values
        colors = _colors_from_values(y)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(x_dt, y, color=colors, edgecolor="black", linewidth=0.5, width=0.8)
        ax.axhline(0, linewidth=1, color="black")
        ax.set_title(f"Sentimiento diario — {tkr}  (y = {metric})")
        ax.set_xlabel("Fecha (UTC)")
        ax.set_ylabel("Sentimiento")
        _setup_date_axis(ax)
        plt.tight_layout()
        plt.show()

    # ---------- 7.2 Portafolio igual-ponderado ----------
    port_equal = ts_df.groupby("date_utc", as_index=False)[metric].mean()
    x_dt = pd.to_datetime(port_equal["date_utc"])
    y = port_equal[metric].values
    colors = _colors_from_values(y)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_dt, y, color=colors, edgecolor="black", linewidth=0.5, width=0.8)
    ax.axhline(0, linewidth=1, color="black")
    ax.set_title(f"Sentimiento diario — Portafolio (igual ponderado)  (y = {metric})")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Sentimiento promedio")
    _setup_date_axis(ax)
    plt.tight_layout()
    plt.show()

    # ---------- 7.3 Portafolio ponderado por titulares (sin FutureWarning) ----------
    port_weighted = _portfolio_weighted_df(ts_df, metric)
    x_dt = pd.to_datetime(port_weighted["date_utc"])
    y = port_weighted[metric].values
    colors = _colors_from_values(y)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x_dt, y, color=colors, edgecolor="black", linewidth=0.5, width=0.8)
    ax.axhline(0, linewidth=1, color="black")
    ax.set_title(f"Sentimiento diario — Portafolio (ponderado por titulares)  (y = {metric})")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Sentimiento promedio ponderado")
    _setup_date_axis(ax)
    plt.tight_layout()
    plt.show()

# ==============================
# MAIN
# ==============================
async def async_main():
    # 1) Scraping
    results = await process_tickers(TICKERS)

    # 2) Resumen rápido a consola
    for r in results:
        print("=" * 100)
        print(f"[{r.ticker}] {r.company} — {len(r.rows)} titulares")
        for i, row in enumerate(r.rows[:8], 1):
            print(f"{i:02d}. ({row['source']}) {row['title']}")
            if row.get("published"):
                print(f"     published: {row['published']}")
        print()

    # 3) DataFrame base
    df_news = to_dataframe(results)
    if df_news.empty:
        print("No se encontraron titulares bajo los filtros. Saliendo.")
        return

    # 4) Sentimiento (CPU)
    print("\nCargando modelo y ejecutando análisis de sentimiento (CPU)...")
    df_scored = run_sentiment(df_news, text_col="title",
                              model_name=HF_MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE)

    # 5) Serie de tiempo
    ts = build_timeseries(df_scored)

    # 6) Vista rápida por ticker (últimos 7 días disponibles)
    print("\nResumen por ticker (últimos 7 días disponibles):")
    for tkr in ts["ticker"].unique():
        sub = ts[ts["ticker"] == tkr].tail(7)
        if sub.empty:
            continue
        print(f"\n[{tkr}]")
        for _, row in sub.iterrows():
            print(f"  {row['date_utc']}  n={int(row['n_headlines'])}  "
                  f"net={row['net_sentiment']:.3f}  "
                  f"pos={row['mean_pos']:.3f}  neg={row['mean_neg']:.3f}  neu={row['mean_neu']:.3f}  "
                  f"share_pos={row['share_pos']:.2%}")

    # 7) Gráficas de barras (se muestran; no se guardan)
    if ts is not None and not ts.empty:
        plot_sentiment_bars(ts, metric="net_sentiment")
    else:
        print("Sin datos de serie de tiempo para graficar (ts vacío).")

def main():
    # Detección de loop activo (Spyder/Jupyter) y ejecución segura
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        running_loop.run_until_complete(async_main())
    else:
        asyncio.run(async_main())

if __name__ == "__main__":
    main()
