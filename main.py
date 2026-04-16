"""
Sepsis Early Warning — FastAPI Backend
Primary model: LSTM + Bio_ClinicalBERT Meta-Fusion
EHR explainability: LSTM gradient-based attribution (torch.autograd)
Run: python -m uvicorn main:app --port 8000
"""

import os, json, warnings, io, base64
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import scipy.sparse as sp

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

ARTIFACTS  = "model_artifacts/"
BERT_CACHE = "bert_cache/"
BERT_NAME  = "emilyalsentzer/Bio_ClinicalBERT"

app = FastAPI(title="Sepsis Early Warning API")

# ══════════════════════════════════════════════════════════════════
# STARTUP — LSTM + small JSON artifacts only (fast)
# ══════════════════════════════════════════════════════════════════

print("Loading LSTM…")
with open(ARTIFACTS + "lstm_config.json") as f:
    cfg = json.load(f)
LSTM_FEAT_COLS = cfg["feat_cols"]
SEQ_LEN        = cfg["seq_len"]
lstm_model     = torch.jit.load(ARTIFACTS + "lstm_model_scripted.pt", map_location="cpu")
lstm_model.eval()
print("  ✅ LSTM ready")

print("Loading meta-model…")
with open(ARTIFACTS + "meta_model.json") as f:
    mm = json.load(f)
meta_model = LogisticRegression()
meta_model.coef_      = np.array(mm["coef"])
meta_model.intercept_ = np.array(mm["intercept"])
meta_model.classes_   = np.array(mm["classes"])
print("  ✅ meta_model")

print("Loading BERT classifier…")
with open(ARTIFACTS + "bert_scaler.json") as f:
    bs = json.load(f)
bert_scaler = StandardScaler()
bert_scaler.mean_          = np.array(bs["mean"])
bert_scaler.scale_         = np.array(bs["scale"])
bert_scaler.var_           = bert_scaler.scale_ ** 2
bert_scaler.n_features_in_ = len(bs["mean"])

with open(ARTIFACTS + "bert_clf.json") as f:
    bc = json.load(f)
bert_clf = LogisticRegression()
bert_clf.coef_      = np.array(bc["coef"])
bert_clf.intercept_ = np.array(bc["intercept"])
bert_clf.classes_   = np.array(bc["classes"])
print("  ✅ bert_clf + bert_scaler")

print("Loading BERT note prior…")
with open(ARTIFACTS + "bert_note_prior.json") as f:
    nd = json.load(f)
mean_train_emb   = np.array(nd["mean_train_emb"], dtype=np.float32)
train_prevalence = float(nd["train_prevalence"])
print("  ✅ bert_note_prior")

print("Loading TF-IDF…")
with open(ARTIFACTS + "tfidf.json") as f:
    td = json.load(f)
tfidf = TfidfVectorizer(
    vocabulary=td["vocabulary"],
    ngram_range=tuple(td["ngram_range"]),
    max_features=td["max_features"],
)
tfidf.idf_ = np.array(td["idf"])
tfidf._tfidf._idf_diag = sp.diags(
    tfidf.idf_, offsets=0,
    shape=(len(tfidf.idf_), len(tfidf.idf_)),
    format="csr", dtype=np.float64,
)
print("  ✅ tfidf")

# BERT neural model — lazy loaded on first prediction
bert_tok      = None
bert_model_hf = None

print("\n🚀 Server ready — open http://localhost:8000")
print("BERT loads on your first CSV upload (~20s first time, instant after).\n")

# ══════════════════════════════════════════════════════════════════
# LAZY LOADER
# ══════════════════════════════════════════════════════════════════

def ensure_bert():
    global bert_tok, bert_model_hf
    if bert_tok is None:
        from transformers import AutoTokenizer, AutoModel
        print("  Loading Bio_ClinicalBERT on first request…")
        os.makedirs(BERT_CACHE, exist_ok=True)
        bert_tok      = AutoTokenizer.from_pretrained(BERT_NAME, cache_dir=BERT_CACHE)
        bert_model_hf = AutoModel.from_pretrained(
            BERT_NAME, cache_dir=BERT_CACHE, torch_dtype=torch.float32
        ).to("cpu")
        bert_model_hf.eval()
        print("  ✅ BERT ready")

# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════

def build_seq(df):
    """Build (SEQ_LEN, n_feat) float32 tensor from CSV."""
    seq = df[[c for c in LSTM_FEAT_COLS if c in df.columns]].fillna(0).values.astype(np.float32)
    if len(seq) < SEQ_LEN:
        seq = np.vstack([
            np.zeros((SEQ_LEN - len(seq), len(LSTM_FEAT_COLS)), dtype=np.float32), seq
        ])
    else:
        seq = seq[-SEQ_LEN:]
    return seq

def infer_lstm(seq_np):
    x = torch.tensor(seq_np).unsqueeze(0)   # (1, T, F)
    with torch.no_grad():
        return float(torch.sigmoid(lstm_model(x)).item())

def infer_bert(note_text):
    if not note_text.strip():
        return train_prevalence, False
    try:
        ensure_bert()
        inputs = bert_tok(
            note_text[:512], return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        with torch.no_grad():
            out = bert_model_hf(**inputs)
        emb   = out.last_hidden_state[:, 0, :].cpu().numpy()
        emb_s = bert_scaler.transform(emb)
        return float(bert_clf.predict_proba(emb_s)[0, 1]), True
    except Exception as e:
        print(f"  BERT error: {e} — using base-rate prior")
        return train_prevalence, False

def infer_meta(lstm_p, bert_p):
    return float(meta_model.predict_proba([[lstm_p, bert_p]])[0, 1])

def infer_tfidf(note_text):
    if not note_text.strip():
        return []
    vec   = tfidf.transform([note_text])
    dense = vec.toarray()[0]
    names = tfidf.get_feature_names_out()
    top   = np.argsort(dense)[-12:][::-1]
    return [{"term": names[i], "weight": round(float(dense[i]), 5)}
            for i in top if dense[i] > 0]

# ══════════════════════════════════════════════════════════════════
# LSTM GRADIENT ATTRIBUTION
# ══════════════════════════════════════════════════════════════════

def lstm_gradient_attribution(seq_np):
    """
    Compute input × gradient attribution for the LSTM.

    For each input dimension (timestep t, feature f):
        attr[t, f] = input[t, f] × d(output) / d(input[t, f])

    This is the standard saliency × input method — exact, fast (~0.5s),
    and interpretable as "how much did this feature at this timestep
    contribute to the LSTM's output in the direction of the output."

    Returns
    -------
    feature_importance : np.ndarray (n_features,)
        Mean absolute attribution across all timesteps per feature.
    timestep_importance : np.ndarray (SEQ_LEN,)
        Mean absolute attribution across all features per timestep.
    attr_matrix : np.ndarray (SEQ_LEN, n_features)
        Full attribution matrix for heatmap.
    """
    # TorchScript models don't support autograd directly on logits,
    # so we compute gradients w.r.t. the sigmoid output
    x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    x.requires_grad_(True)

    # Forward pass — need grad so can't use no_grad
    logit  = lstm_model(x)                          # (1, 1)
    output = torch.sigmoid(logit)                   # (1, 1)
    output.backward()                               # compute grads

    grads  = x.grad.squeeze(0).detach().numpy()     # (T, F)
    inputs = seq_np                                 # (T, F)

    # Input × gradient (signed attribution)
    attr = inputs * grads                            # (T, F)

    feature_importance  = np.abs(attr).mean(axis=0) # (F,)  mean over time
    timestep_importance = np.abs(attr).mean(axis=1) # (T,)  mean over features

    return feature_importance, timestep_importance, attr

# ══════════════════════════════════════════════════════════════════
# META-MODEL ATTRIBUTION (logistic regression — analytical)
# ══════════════════════════════════════════════════════════════════

def compute_meta_attribution(lstm_p, bert_p):
    coefs     = meta_model.coef_[0]
    intercept = float(meta_model.intercept_[0])
    names     = ["LSTM (EHR time-series)", "ClinicalBERT (clinical note)"]
    values    = [lstm_p, bert_p]
    contribs  = [float(coefs[i]) * values[i] for i in range(2)]
    return [
        {"feature": names[i], "value": round(values[i], 4),
         "contribution": round(contribs[i], 5)}
        for i in range(2)
    ], intercept

# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════

BG, GRID     = "#0a0e17", "#1e2433"
TEXT, MUTED  = "#e2e8f8", "#7a8499"
RED, GREEN   = "#ef4b6c", "#34d399"
BLUE, PURPLE = "#60a5fa", "#a78bfa"
AMB          = "#f59e0b"

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def setup_ax(fig, ax):
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

def make_lstm_attribution_chart(feat_imp, ts_imp, attr_matrix):
    """
    Three-panel chart:
    1. Feature importance bar chart (mean |attribution| per feature)
    2. Timestep importance line chart (which hours mattered most)
    3. Attribution heatmap (T × F)
    """
    n_feat = len(LSTM_FEAT_COLS)
    n_time = len(ts_imp)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1, 1],
                            hspace=0.45, wspace=0.35)

    ax_feat = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    for ax in [ax_feat, ax_time, ax_heat]:
        setup_ax(fig, ax)

    # ── 1. Feature importance ──────────────────────────────────────
    sorted_idx  = np.argsort(feat_imp)
    sorted_imp  = feat_imp[sorted_idx]
    sorted_name = [LSTM_FEAT_COLS[i] for i in sorted_idx]
    colors      = [RED if v > np.median(feat_imp) else BLUE for v in sorted_imp]

    ax_feat.barh(sorted_name, sorted_imp, color=colors, height=0.6, edgecolor="none")
    ax_feat.set_xlabel("Mean |attribution| across timesteps")
    ax_feat.set_title("LSTM — Feature Importance\n(Input × Gradient)", pad=8)
    ax_feat.legend(
        handles=[mpatches.Patch(color=RED,  label="High importance"),
                 mpatches.Patch(color=BLUE, label="Low importance")],
        fontsize=7.5, framealpha=0.15, facecolor=BG,
        labelcolor=TEXT, edgecolor=GRID,
    )

    # ── 2. Timestep importance ─────────────────────────────────────
    hours = [f"h{i+1}" for i in range(n_time)]
    ax_time.plot(range(n_time), ts_imp, color=BLUE, linewidth=1.5)
    ax_time.fill_between(range(n_time), ts_imp, alpha=0.2, color=BLUE)
    peak = int(np.argmax(ts_imp))
    ax_time.axvline(peak, color=AMB, linewidth=1, linestyle="--",
                    label=f"Peak: h{peak+1}")
    ax_time.set_xticks(range(0, n_time, 3))
    ax_time.set_xticklabels([f"h{i+1}" for i in range(0, n_time, 3)],
                             rotation=45, fontsize=7)
    ax_time.set_xlabel("Observation hour (1 = oldest, 18 = most recent)")
    ax_time.set_ylabel("Mean |attribution|")
    ax_time.set_title("LSTM — Timestep Importance\nWhich hours drove the prediction", pad=8)
    ax_time.legend(fontsize=7.5, framealpha=0.15, facecolor=BG,
                   labelcolor=TEXT, edgecolor=GRID)

    # ── 3. Heatmap (T × F) ────────────────────────────────────────
    # Clip extreme values for visual clarity
    vmax = np.percentile(np.abs(attr_matrix), 95)
    vmin = -vmax

    im = ax_heat.imshow(
        attr_matrix.T,          # (F, T) so features on y-axis
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(n_feat))
    ax_heat.set_yticklabels(LSTM_FEAT_COLS, fontsize=7.5)
    ax_heat.set_xticks(range(0, n_time, 3))
    ax_heat.set_xticklabels([f"h{i+1}" for i in range(0, n_time, 3)],
                              rotation=0, fontsize=7.5)
    ax_heat.set_xlabel("Observation hour")
    ax_heat.set_title(
        "LSTM — Attribution Heatmap  (red = increases risk, blue = decreases risk)\n"
        "Row = vital/lab feature   |   Column = hour in 18h observation window",
        pad=8,
    )

    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("Attribution (input × gradient)", color=MUTED, fontsize=7.5)

    b64 = fig_to_b64(fig)
    plt.close()
    return b64

def make_meta_attribution_chart(meta_data, intercept):
    """Modality log-odds contributions + weight split."""
    names    = [d["feature"]      for d in meta_data]
    contribs = [d["contribution"] for d in meta_data]
    values   = [d["value"]        for d in meta_data]
    colors   = [BLUE, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax in axes:
        setup_ax(fig, ax)

    bars = axes[0].barh(names, contribs, color=colors, height=0.45, edgecolor="none")
    axes[0].axvline(0, color=GRID, linewidth=1)
    axes[0].set_xlabel("Log-odds contribution")
    axes[0].set_title("Meta-model — Modality Attribution")
    for bar, v, val in zip(bars, contribs, values):
        axes[0].text(
            v + (0.02 if v >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}  (score={val:.3f})",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8.5, color=TEXT,
        )

    coefs   = meta_model.coef_[0]
    total_w = abs(coefs[0]) + abs(coefs[1])
    w_pcts  = [abs(coefs[0]) / total_w * 100, abs(coefs[1]) / total_w * 100]
    lstm_c  = values[0] * abs(coefs[0])
    bert_c  = values[1] * abs(coefs[1])
    total_c = lstm_c + bert_c + 1e-9
    c_pcts  = [lstm_c / total_c * 100, bert_c / total_c * 100]

    x  = np.arange(2)
    w  = 0.35
    b1 = axes[1].bar(x - w/2, w_pcts, w, color=colors, alpha=0.5,
                     label="Learned weight", edgecolor="none")
    b2 = axes[1].bar(x + w/2, c_pcts,  w, color=colors, alpha=1.0,
                     label="This patient",   edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["LSTM", "BERT"], color=TEXT)
    axes[1].set_ylabel("(%)")
    axes[1].set_title("Weight vs Contribution")
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=7.5, framealpha=0.2, facecolor=BG,
                   labelcolor=TEXT, edgecolor=GRID)
    for bars_, pcts in [(b1, w_pcts), (b2, c_pcts)]:
        for bar, v in zip(bars_, pcts):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1.5,
                         f"{v:.1f}%", ha="center", fontsize=7.5, color=MUTED)

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close()
    return b64

# ══════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Extract note
    note_text = ""
    if "note_text" in df.columns:
        notes = df["note_text"].dropna().astype(str)
        notes = notes[notes.str.strip() != ""]
        if len(notes):
            note_text = notes.iloc[0]

    # Build sequence
    seq_np = build_seq(df)

    # Run pipeline
    lstm_p           = infer_lstm(seq_np)
    bert_p, has_note = infer_bert(note_text)
    fusion_p         = infer_meta(lstm_p, bert_p)

    # LSTM gradient attribution
    feat_imp, ts_imp, attr_matrix = lstm_gradient_attribution(seq_np)

    # Meta attribution
    meta_data, intercept = compute_meta_attribution(lstm_p, bert_p)

    # TF-IDF
    tfidf_terms = infer_tfidf(note_text) if has_note else []

    # Vitals
    last   = df.iloc[-1]
    vitals = {
        col: (round(float(last[col]), 2)
              if col in df.columns and pd.notna(last[col]) else None)
        for col in ["HR","SBP","DBP","MAP","SpO2","Temp","RR",
                    "lactate","wbc","gcs","vaso_dose","on_vent"]
    }

    # Charts
    lstm_chart = make_lstm_attribution_chart(feat_imp, ts_imp, attr_matrix)
    meta_chart = make_meta_attribution_chart(meta_data, intercept)

    coefs = meta_model.coef_[0]
    risk  = "HIGH" if fusion_p >= 0.6 else "MODERATE" if fusion_p >= 0.35 else "LOW"

    # Feature importance as list for frontend table
    feat_sorted = sorted(
        [{"feature": LSTM_FEAT_COLS[i], "importance": round(float(feat_imp[i]), 5)}
         for i in range(len(LSTM_FEAT_COLS))],
        key=lambda x: x["importance"], reverse=True,
    )

    return {
        "filename":         file.filename,
        "fusion_prob":      round(fusion_p, 4),
        "lstm_prob":        round(lstm_p,   4),
        "bert_prob":        round(bert_p,   4),
        "risk_level":       risk,
        "has_note":         has_note,
        "note_preview":     note_text[:400] if has_note else "",
        "train_prevalence": round(train_prevalence, 4),
        "meta_coefs": {
            "lstm_weight": round(float(coefs[0]), 4),
            "bert_weight": round(float(coefs[1]), 4),
            "intercept":   round(float(meta_model.intercept_[0]), 4),
        },
        "meta_attributions":  meta_data,
        "feat_importance":    feat_sorted,
        "peak_timestep":      int(np.argmax(ts_imp)) + 1,
        "vitals":             vitals,
        "tfidf_terms":        tfidf_terms,
        "charts": {
            "lstm_attribution": lstm_chart,   # 3-panel: bar + line + heatmap
            "meta_attribution": meta_chart,   # modality log-odds + weight split
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "models": ["lstm", "bert", "meta"]}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")