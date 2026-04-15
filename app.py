"""
Streamlit GUI for LSTM Paper — Hochreiter & Schmidhuber (1997)
Interactive demo of all experiments from the paper.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Paper — Hochreiter & Schmidhuber 1997",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.paper-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #58a6ff;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.paper-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0 0 8px 0;
}
.paper-authors {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
}
.paper-journal {
    color: #3fb950;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    margin: 6px 0 0 0;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-label {
    color: #8b949e;
    font-size: 0.8rem;
    margin-top: 4px;
}

.equation-box {
    background: #1c2128;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 14px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: #e6edf3;
    margin: 10px 0;
}

.result-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-success { background: #1a4731; color: #3fb950; border: 1px solid #3fb950; }
.badge-fail { background: #3d1f1f; color: #f85149; border: 1px solid #f85149; }
.badge-warn { background: #3d2f0f; color: #d29922; border: 1px solid #d29922; }

.sidebar-section {
    background: #1c2128;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Models ─────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, task="binary"):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.task == "binary":
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze()
        elif self.task == "regression":
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze()
        else:  # multiclass
            return self.fc(out[:, -1, :])


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, task="binary"):
        super().__init__()
        self.task = task
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.task in ("binary", "regression"):
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze()
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, task="binary"):
        super().__init__()
        self.task = task
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        if self.task in ("binary", "regression"):
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze()
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Data generators ────────────────────────────────────────────────────────────
@st.cache_data
def generate_exp2a(p=10, n=1000):
    vocab = p + 1
    x_idx, y_idx = p - 1, p
    X, y = [], []
    for _ in range(n):
        cls = np.random.randint(0, 2)
        fi = y_idx if cls else x_idx
        idxs = [fi] + list(range(p - 2)) + [fi]
        seq = np.zeros((len(idxs), vocab))
        for t, i in enumerate(idxs): seq[t, i] = 1.0
        X.append(seq); y.append(float(cls))
    return np.array(X), np.array(y)


@st.cache_data
def generate_adding(T=100, n=1000):
    X = np.zeros((n, T, 2)); y = np.zeros(n)
    for i in range(n):
        vals = np.random.uniform(-1, 1, T)
        markers = np.zeros(T)
        markers[0] = markers[-1] = -1
        m1 = np.random.randint(1, min(10, T // 2))
        m2 = np.random.randint(T // 2, T - 1)
        markers[m1] = markers[m2] = 1.0
        X1 = 0.0 if m1 == 0 else vals[m1]
        X[i, :, 0] = vals; X[i, :, 1] = markers
        y[i] = 0.5 + (X1 + vals[m2]) / 4.0
    return X, y


@st.cache_data
def generate_temporal(n=1000, seq_len=100):
    vocab_map = {'E':0,'B':1,'a':2,'b':3,'c':4,'d':5,'X':6,'Y':7}
    distractors = ['a','b','c','d']
    class_map = {('X','X'):0,('X','Y'):1,('Y','X'):2,('Y','Y'):3}
    X_d = np.zeros((n, seq_len, 8)); y_d = np.zeros(n, dtype=int)
    for i in range(n):
        t1, t2 = np.random.randint(10,20), np.random.randint(50,60)
        s1, s2 = np.random.choice(['X','Y']), np.random.choice(['X','Y'])
        seq = ['E'] + [np.random.choice(distractors) for _ in range(seq_len-2)] + ['B']
        seq[t1] = s1; seq[t2] = s2
        for t, ch in enumerate(seq[:seq_len]):
            X_d[i, t, vocab_map[ch]] = 1.0
        y_d[i] = class_map[(s1, s2)]
    return X_d, y_d


# ── Training helper ────────────────────────────────────────────────────────────
def train_step(model, X, y, task, epochs=30, lr=0.005, batch=64):
    Xt = torch.FloatTensor(X).to(device)
    yt = torch.LongTensor(y).to(device) if task == "multiclass" else torch.FloatTensor(y).to(device)
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss() if task == "multiclass" else nn.MSELoss() if task == "regression" else nn.BCELoss()
    losses, accs = [], []
    for _ in range(epochs):
        model.train(); ep = 0
        for xb, yb in loader:
            opt.zero_grad(); pred = model(xb)
            loss = crit(pred, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep += loss.item()
        model.eval()
        with torch.no_grad():
            p = model(Xt)
            if task == "multiclass":
                acc = (p.argmax(1).cpu().numpy() == y).mean()
            elif task == "binary":
                acc = ((p.cpu().numpy() > 0.5) == y).mean()
            else:
                acc = 1 - mean_squared_error(y, p.cpu().numpy())
        losses.append(ep / len(loader)); accs.append(acc)
    return model, losses, accs


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 LSTM Paper GUI")
    st.markdown("**Hochreiter & Schmidhuber, 1997**")
    st.divider()

    page = st.radio("📌 Navigate", [
        "🏠 Paper Overview",
        "📊 Exp 2a — Long Time Lag",
        "➕ Exp 4 — Adding Problem",
        "🕐 Exp 6 — Temporal Order",
        "⚡ Live Demo",
        "📐 Architecture Viz"
    ])

    st.divider()
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**⚙️ Hyperparameters**")
    hidden_size = st.slider("Hidden Size", 8, 64, 32, 8)
    epochs = st.slider("Epochs", 10, 100, 40, 10)
    lr = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], value=0.005)
    n_samples = st.select_slider("Samples", [500, 1000, 2000], value=1000)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**📌 Quick Facts**")
    st.caption("• Paper: Neural Computation 9(8), 1997")
    st.caption("• CEC ensures constant error flow")
    st.caption("• Bridges time lags > 1000 steps")
    st.caption(f"• Device: {'GPU 🔥' if device.type == 'cuda' else 'CPU 💻'}")


# ── Pages ──────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Paper Overview":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="paper-header">
        <p class="paper-title">LONG SHORT-TERM MEMORY</p>
        <p class="paper-authors">Sepp Hochreiter &nbsp;·&nbsp; Jürgen Schmidhuber</p>
        <p class="paper-journal">Neural Computation 9(8):1735–1780 · 1997</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📖 Abstract Summary")
        st.info("""
        LSTM solves the **vanishing/exploding gradient problem** in recurrent networks
        by introducing **memory cells** with gated access — allowing error signals to
        flow unchanged for 1000+ time steps via the **Constant Error Carousel (CEC)**.
        """)

        st.markdown("### 🔑 Key Equations")
        st.markdown('<div class="equation-box">Input Gate: &nbsp; y^{in}(t) = f(W_in · [x,h] + b_in)</div>', unsafe_allow_html=True)
        st.markdown('<div class="equation-box">Output Gate: &nbsp; y^{out}(t) = f(W_out · [x,h] + b_out)</div>', unsafe_allow_html=True)
        st.markdown('<div class="equation-box">CEC State: &nbsp; s_c(t) = s_c(t-1) + y^{in}(t) · g(W_c · [x,h])</div>', unsafe_allow_html=True)
        st.markdown('<div class="equation-box">Cell Out: &nbsp; y^c(t) = y^{out}(t) · h(s_c(t))</div>', unsafe_allow_html=True)
        st.markdown('<div class="equation-box">Constant Error: &nbsp; ∂s_c(t)/∂s_c(t-1) ≈ 1 &nbsp; [Eq.35]</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧪 Paper Experiments")
        exps = {
            "Exp 1": ("Embedded Reber Grammar", "✅"),
            "Exp 2a": ("Noise-free Long Time Lag", "✅"),
            "Exp 2b": ("No Local Regularities", "✅"),
            "Exp 2c": ("Very Long Lags (1000 steps)", "✅"),
            "Exp 3a": ("2-Sequence Problem", "✅"),
            "Exp 4": ("Adding Problem", "✅"),
            "Exp 5": ("Multiplication Problem", "✅"),
            "Exp 6a": ("Temporal Order (2 symbols)", "✅"),
            "Exp 6b": ("Temporal Order (3 symbols)", "✅"),
        }
        for exp, (desc, badge) in exps.items():
            st.markdown(f"**{exp}** — {desc} {badge}")

        st.markdown("### 📊 Paper Results Table 2")
        paper_results = pd.DataFrame({
            "Method": ["RTRL", "BPTT", "CH", "**LSTM**"],
            "Delay p": [10, 100, 100, 100],
            "Success %": [0, 0, 33, "**100**"],
            "Sequences": [">5M", ">5M", "32,400", "**5,040**"]
        })
        st.table(paper_results)

    st.markdown("### 🆚 Model Architecture Comparison")
    cols = st.columns(4)
    models_info = [
        ("LSTM", "#58a6ff", "Constant Error Carousel\nInput + Output Gates\nO(W) complexity"),
        ("Vanilla RNN", "#f85149", "Simple recurrence\nNo gating\nVanishing gradient"),
        ("GRU", "#d29922", "Reset + Update gates\nNo output gate\nPost-1997"),
        ("BiLSTM", "#3fb950", "Bidirectional LSTM\nForward + Backward\nBetter context"),
    ]
    for col, (name, color, desc) in zip(cols, models_info):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}40">
                <div class="metric-val" style="color:{color};font-size:1.1rem">{name}</div>
                <div class="metric-label" style="white-space:pre-line">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exp 2a — Long Time Lag":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Experiment 2a: Noise-Free Long Time Lag")
    st.markdown("""
    **Section 5.2** — Two sequences: `(y, a1,...,ap-1, y)` and `(x, a1,...,ap-1, x)`.
    The network must **remember the first symbol** for `p` steps to predict the last.
    """)

    p_val = st.slider("Time Lag (p)", 4, 30, 10, 2)
    models_to_run = st.multiselect("Models to Compare", ["LSTM", "RNN", "GRU"], default=["LSTM", "RNN", "GRU"])

    if st.button("▶ Run Experiment 2a", type="primary"):
        with st.spinner("Generating data & training models..."):
            X_tr, y_tr = generate_exp2a(p=p_val, n=n_samples)
            X_te, y_te = generate_exp2a(p=p_val, n=500)
            input_sz = X_tr.shape[2]

            results = {}
            progress = st.progress(0)

            for idx, mname in enumerate(models_to_run):
                if mname == "LSTM":
                    m = LSTMModel(input_sz, hidden_size, task="binary")
                elif mname == "RNN":
                    m = VanillaRNN(input_sz, hidden_size, task="binary")
                else:
                    m = GRUModel(input_sz, hidden_size, task="binary")

                m, losses, accs = train_step(m, X_tr, y_tr, "binary", epochs, lr)
                m.eval()
                with torch.no_grad():
                    preds = m(torch.FloatTensor(X_te).to(device)).cpu().numpy()
                test_acc = ((preds > 0.5) == y_te).mean()
                results[mname] = {"losses": losses, "accs": accs, "test_acc": test_acc}
                progress.progress((idx + 1) / len(models_to_run))

        # Metrics row
        mcols = st.columns(len(models_to_run))
        colors = {"LSTM": "#58a6ff", "RNN": "#f85149", "GRU": "#3fb950"}
        for col, mname in zip(mcols, models_to_run):
            acc = results[mname]["test_acc"]
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{colors[mname]}">
                    <div class="metric-val" style="color:{colors[mname]}">{acc*100:.1f}%</div>
                    <div class="metric-label">{mname} Test Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
        fig.suptitle(f"Experiment 2a: Time Lag p={p_val}", color="white", fontsize=13, fontweight="bold")
        cmap = {"LSTM": "#58a6ff", "RNN": "#f85149", "GRU": "#3fb950"}
        ls_map = {"LSTM": "-", "RNN": "--", "GRU": "-."}

        for mname, res in results.items():
            axes[0].plot(res["losses"], color=cmap[mname], ls=ls_map[mname], lw=2.5, label=mname)
            axes[1].plot(res["accs"], color=cmap[mname], ls=ls_map[mname], lw=2.5, label=mname)

        for ax in axes:
            ax.set_facecolor("#161b22"); ax.grid(True, alpha=0.2, color="white")
            ax.tick_params(colors="white"); ax.legend(facecolor="#161b22", labelcolor="white")
            for spine in ax.spines.values(): spine.set_color("#30363d")

        axes[0].set_title("BCE Loss", color="white"); axes[0].set_xlabel("Epoch", color="white")
        axes[1].set_title("Accuracy", color="white"); axes[1].set_xlabel("Epoch", color="white")
        axes[1].axhline(0.5, color="#8b949e", ls=":", lw=1, label="Chance")
        axes[0].set_ylabel("Loss", color="white"); axes[1].set_ylabel("Accuracy", color="white")

        st.pyplot(fig)
        plt.close()

        # Paper comparison
        st.markdown("### 📋 Paper Table 2 Comparison")
        comp_df = pd.DataFrame({
            "Method": ["RTRL (paper)", "BPTT (paper)", "LSTM (paper)", f"LSTM (this run, p={p_val})"] + [f"{m} (this run)" for m in models_to_run if m != "LSTM"],
            "Delay": [4, 100, 100, p_val] + [p_val]*(len(models_to_run)-1),
            "Success %": [0, 0, 100, f"{results.get('LSTM',{}).get('test_acc',0)*100:.1f}"] + [f"{results[m]['test_acc']*100:.1f}" for m in models_to_run if m != "LSTM"],
        })
        st.dataframe(comp_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "➕ Exp 4 — Adding Problem":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("## ➕ Experiment 4: Adding Problem")
    st.markdown("""
    **Section 5.4** — Each input is `(value, marker)`. Exactly **two pairs are marked**.
    The network must output `0.5 + (X1 + X2) / 4.0` — the scaled sum of marked values.
    """)

    st.markdown('<div class="equation-box">Target: 0.5 + (X1 + X2) / 4.0 &nbsp; → &nbsp; scales X1+X2 ∈ [-2,2] to [0,1]</div>', unsafe_allow_html=True)

    T_val = st.slider("Sequence Length T (min lag = T/2)", 20, 200, 100, 20)
    run_rnn = st.checkbox("Also run Vanilla RNN (slower)", value=True)

    if st.button("▶ Run Adding Problem", type="primary"):
        with st.spinner("Training on Adding Problem..."):
            X_tr, y_tr = generate_adding(T=T_val, n=n_samples)
            X_te, y_te = generate_adding(T=T_val, n=500)

            lstm_m = LSTMModel(2, hidden_size, output_size=1, task="regression")
            lstm_m, lstm_l, _ = train_step(lstm_m, X_tr, y_tr, "regression", epochs, lr, batch=128)

            lstm_m.eval()
            with torch.no_grad():
                lstm_p = lstm_m(torch.FloatTensor(X_te).to(device)).cpu().numpy()
            lstm_mse = mean_squared_error(y_te, lstm_p)
            lstm_wrong = (np.abs(lstm_p - y_te) > 0.04).sum()

            rnn_l, rnn_p, rnn_mse, rnn_wrong = None, None, None, None
            if run_rnn:
                rnn_m = VanillaRNN(2, hidden_size, task="regression")
                rnn_m, rnn_l, _ = train_step(rnn_m, X_tr, y_tr, "regression", epochs, lr, batch=128)
                rnn_m.eval()
                with torch.no_grad():
                    rnn_p = rnn_m(torch.FloatTensor(X_te).to(device)).cpu().numpy()
                rnn_mse = mean_squared_error(y_te, rnn_p)
                rnn_wrong = (np.abs(rnn_p - y_te) > 0.04).sum()

        # Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card" style="border-color:#58a6ff">
                <div class="metric-val">{lstm_mse:.5f}</div>
                <div class="metric-label">LSTM MSE</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card" style="border-color:#{'3fb950' if lstm_wrong < 20 else 'f85149'}">
                <div class="metric-val">{lstm_wrong}/500</div>
                <div class="metric-label">LSTM Wrong (threshold 0.04)</div></div>""", unsafe_allow_html=True)
        with c3:
            min_lag = T_val // 2
            st.markdown(f"""<div class="metric-card" style="border-color:#d29922">
                <div class="metric-val">{min_lag}</div>
                <div class="metric-label">Min Time Lag</div></div>""", unsafe_allow_html=True)

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d1117")
        fig.suptitle(f"Experiment 4: Adding Problem (T={T_val}, min lag={T_val//2})", color="white", fontsize=13, fontweight="bold")

        axes[0].plot(lstm_l, color="#58a6ff", lw=2.5, label="LSTM")
        if rnn_l: axes[0].plot(rnn_l, color="#f85149", lw=2.5, ls="--", label="RNN")
        axes[0].axhline(0.01, color="#3fb950", ls=":", lw=1.5, label="Paper target")

        idx_s = np.random.choice(len(y_te), min(150, len(y_te)), replace=False)
        axes[1].scatter(y_te[idx_s], lstm_p[idx_s], c="#58a6ff", alpha=0.6, s=25, label="LSTM")
        if rnn_p is not None:
            axes[1].scatter(y_te[idx_s], rnn_p[idx_s], c="#f85149", alpha=0.4, s=15, marker="x", label="RNN")
        axes[1].plot([0, 1], [0, 1], "w--", lw=1.5, label="Perfect")

        errs = np.abs(lstm_p - y_te)
        axes[2].hist(errs, bins=25, color="#58a6ff", alpha=0.8, label=f"LSTM (wrong={lstm_wrong})")
        if rnn_p is not None:
            axes[2].hist(np.abs(rnn_p - y_te), bins=25, color="#f85149", alpha=0.5, label=f"RNN (wrong={rnn_wrong})")
        axes[2].axvline(0.04, color="white", ls="--", lw=2, label="Threshold (0.04)")

        titles = ["Training MSE Loss", "Predicted vs True", "Error Distribution"]
        xlabels = ["Epoch", "True Value", "Absolute Error"]
        for i, ax in enumerate(axes):
            ax.set_facecolor("#161b22"); ax.grid(True, alpha=0.2, color="white")
            ax.tick_params(colors="white"); ax.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
            for spine in ax.spines.values(): spine.set_color("#30363d")
            ax.set_title(titles[i], color="white"); ax.set_xlabel(xlabels[i], color="white")

        axes[0].set_ylabel("MSE", color="white")
        axes[1].set_ylabel("Predicted Value", color="white")
        axes[2].set_ylabel("Count", color="white")

        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕐 Exp 6 — Temporal Order":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("## 🕐 Experiment 6: Temporal Order")
    st.markdown("""
    **Section 5.6** — Classify sequences by the temporal order of **two widely separated symbols** (X, Y).
    Rules from paper: X,X→Q | X,Y→R | Y,X→S | Y,Y→U
    """)

    st.markdown("""
    | First Symbol | Second Symbol | Class |
    |:---:|:---:|:---:|
    | X | X | Q (0) |
    | X | Y | R (1) |
    | Y | X | S (2) |
    | Y | Y | U (3) |
    """)

    seq_len = st.slider("Sequence Length", 50, 150, 100, 10)
    run_bilstm = st.checkbox("Also run BiLSTM", value=True)

    if st.button("▶ Run Temporal Order", type="primary"):
        with st.spinner("Generating temporal sequences & training..."):
            X_tr, y_tr = generate_temporal(n=n_samples, seq_len=seq_len)
            X_te, y_te = generate_temporal(n=400, seq_len=seq_len)

            lstm_m = LSTMModel(8, hidden_size, output_size=4, task="multiclass")
            lstm_m, lstm_l, lstm_a = train_step(lstm_m, X_tr, y_tr, "multiclass", epochs, lr)
            lstm_m.eval()
            with torch.no_grad():
                lstm_p = lstm_m(torch.FloatTensor(X_te).to(device)).argmax(1).cpu().numpy()
            lstm_acc = (lstm_p == y_te).mean()

            bilstm_l, bilstm_a, bilstm_p, bilstm_acc = None, None, None, None
            if run_bilstm:
                bi_m = BiLSTMModel(8, hidden_size // 2, output_size=4)
                bi_m, bilstm_l, bilstm_a = train_step(bi_m, X_tr, y_tr, "multiclass", epochs, lr)
                bi_m.eval()
                with torch.no_grad():
                    bilstm_p = bi_m(torch.FloatTensor(X_te).to(device)).argmax(1).cpu().numpy()
                bilstm_acc = (bilstm_p == y_te).mean()

        # Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card" style="border-color:#58a6ff">
                <div class="metric-val">{lstm_acc*100:.1f}%</div>
                <div class="metric-label">LSTM Test Accuracy</div></div>""", unsafe_allow_html=True)
        with c2:
            if bilstm_acc is not None:
                st.markdown(f"""<div class="metric-card" style="border-color:#3fb950">
                    <div class="metric-val">{bilstm_acc*100:.1f}%</div>
                    <div class="metric-label">BiLSTM Test Accuracy</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card" style="border-color:#d29922">
                <div class="metric-val">25.0%</div>
                <div class="metric-label">Chance Level (4 classes)</div></div>""", unsafe_allow_html=True)

        # Plots
        ncols = 3 if run_bilstm else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), facecolor="#0d1117")
        fig.suptitle("Experiment 6a: Temporal Order", color="white", fontsize=13, fontweight="bold")

        axes[0].plot(lstm_a, color="#58a6ff", lw=2.5, label="LSTM")
        if bilstm_a: axes[0].plot(bilstm_a, color="#3fb950", lw=2.5, ls="--", label="BiLSTM")
        axes[0].axhline(0.25, color="#8b949e", ls=":", lw=1, label="Chance")

        cm = confusion_matrix(y_te, lstm_p)
        class_names = ["XX→Q", "XY→R", "YX→S", "YY→U"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#30363d")
        axes[1].set_title(f"LSTM Confusion (Acc={lstm_acc:.3f})", color="white")

        if run_bilstm and bilstm_p is not None and ncols == 3:
            cm2 = confusion_matrix(y_te, bilstm_p)
            sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axes[2],
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.5, linecolor="#30363d")
            axes[2].set_title(f"BiLSTM Confusion (Acc={bilstm_acc:.3f})", color="white")

        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_color("#30363d")

        axes[0].set_ylabel("Accuracy", color="white")
        axes[0].set_xlabel("Epoch", color="white")
        axes[0].set_title("Training Accuracy", color="white")
        axes[0].grid(True, alpha=0.2, color="white")
        axes[0].legend(facecolor="#161b22", labelcolor="white")

        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Live Demo":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("## ⚡ Live Demo — Test on Your Own Input")

    demo_type = st.selectbox("Select Demo Task", [
        "🔢 Adding Problem — Custom Sequence",
        "📝 Temporal Symbol Classification",
    ])

    if "🔢" in demo_type:
        st.markdown("### Enter a Custom Sequence for the Adding Problem")
        st.markdown("Enter space-separated `value,marker` pairs. Marker: `1`=marked, `0`=normal, `-1`=edge.")
        st.info("Example: `0.5,−1 0.3,0 0.8,1 −0.2,0 0.6,1 0.1,0 0.4,−1` — sum marked values (0.8 + 0.6 = 1.4 → 0.5 + 1.4/4 = 0.85)")

        example_seq = "0.5,-1 0.3,0 0.8,1 -0.2,0 0.6,1 0.1,0 0.4,-1"
        user_input = st.text_area("Sequence (value,marker pairs):", value=example_seq, height=80)

        if st.button("🔮 Predict"):
            try:
                pairs = [tuple(map(float, p.strip().split(","))) for p in user_input.strip().split()]
                seq = np.array([[v, m] for v, m in pairs])
                marked = [(i, v) for i, (v, m) in enumerate(pairs) if m == 1.0]
                X1 = marked[0][1] if marked else 0
                X2 = marked[1][1] if len(marked) > 1 else 0
                true_target = 0.5 + (X1 + X2) / 4.0

                # Quick model
                X_tr, y_tr = generate_adding(T=max(10, len(seq)), n=500)
                m = LSTMModel(2, 16, output_size=1, task="regression")
                m, _, _ = train_step(m, X_tr, y_tr, "regression", epochs=20, lr=0.01, batch=64)
                m.eval()
                inp = torch.FloatTensor(seq).unsqueeze(0).to(device)
                # Pad/truncate to training length
                T_train = X_tr.shape[1]
                inp_padded = torch.zeros(1, T_train, 2).to(device)
                inp_padded[0, :min(len(seq), T_train)] = inp[0, :min(len(seq), T_train)]
                with torch.no_grad():
                    pred = m(inp_padded).item()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("True Target", f"{true_target:.4f}")
                with c2:
                    st.metric("LSTM Prediction", f"{pred:.4f}")
                with c3:
                    st.metric("Error", f"{abs(pred - true_target):.4f}",
                             delta_color="inverse")

                st.markdown(f"**Marked values:** {[f'pos {i}: {v:.2f}' for i, v in marked]}")
                st.markdown(f"**Sum formula:** 0.5 + ({X1:.2f} + {X2:.2f}) / 4.0 = **{true_target:.4f}**")

                # Visualize the sequence
                fig, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor="#0d1117")
                axes[0].bar(range(len(pairs)), [v for v, m in pairs], color=[
                    "#58a6ff" if m == 1.0 else "#30363d" for v, m in pairs
                ], edgecolor="#8b949e")
                axes[0].set_title("Input Sequence (blue = marked)", color="white")
                axes[0].set_ylabel("Value", color="white")

                axes[1].axhline(true_target, color="#3fb950", lw=2.5, label=f"True target: {true_target:.4f}")
                axes[1].axhline(pred, color="#58a6ff", lw=2.5, ls="--", label=f"LSTM pred: {pred:.4f}")
                axes[1].set_ylim(0, 1); axes[1].set_title("Prediction vs Target", color="white")
                axes[1].legend(facecolor="#161b22", labelcolor="white")

                for ax in axes:
                    ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
                    ax.grid(True, alpha=0.2, color="white")
                    for spine in ax.spines.values(): spine.set_color("#30363d")

                st.pyplot(fig); plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.markdown("### Temporal Order Classifier")
        st.markdown("Enter a sequence of symbols. Available: `E`, `B`, `a`, `b`, `c`, `d`, `X`, `Y`")
        st.info("Example: E a b X c d a Y b B — has X then Y, should classify as X,Y→R (class 1)")

        user_seq = st.text_input("Sequence (space separated):", value="E a b X c d a Y b B")
        if st.button("🔮 Classify Temporal Order"):
            vocab_map = {'E': 0, 'B': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'X': 6, 'Y': 7}
            try:
                tokens = user_seq.strip().upper().split()
                xs_ys = [t for t in tokens if t in ('X', 'Y')]
                if len(xs_ys) < 2:
                    st.warning("Please include at least 2 X or Y symbols.")
                else:
                    s1, s2 = xs_ys[0], xs_ys[1]
                    class_map = {('X', 'X'): ("Q", 0), ('X', 'Y'): ("R", 1),
                                 ('Y', 'X'): ("S", 2), ('Y', 'Y'): ("U", 3)}
                    true_class_name, true_class_idx = class_map[(s1, s2)]

                    seq = np.zeros((len(tokens), 8))
                    for t, tok in enumerate(tokens):
                        if tok.lower() in ('x', 'y'):
                            idx = vocab_map[tok.upper()]
                        else:
                            idx = vocab_map.get(tok.lower(), 2)
                        seq[t, idx] = 1.0

                    # Quick train
                    X_tr, y_tr = generate_temporal(n=500)
                    m = LSTMModel(8, 16, output_size=4, task="multiclass")
                    m, _, _ = train_step(m, X_tr, y_tr, "multiclass", epochs=20, lr=0.01)
                    m.eval()
                    T_ref = X_tr.shape[1]
                    inp = torch.zeros(1, T_ref, 8).to(device)
                    inp[0, :min(len(seq), T_ref)] = torch.FloatTensor(seq[:min(len(seq), T_ref)])
                    with torch.no_grad():
                        logits = m(inp)
                        pred_class = logits.argmax(1).item()
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    class_names = ["Q (X,X)", "R (X,Y)", "S (Y,X)", "U (Y,Y)"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**True Class:** {true_class_name} ({s1},{s2}) → class {true_class_idx}")
                        st.info(f"**LSTM Predicted:** {class_names[pred_class]}")
                        for i, (cn, p) in enumerate(zip(class_names, probs)):
                            st.progress(float(p), text=f"{cn}: {p*100:.1f}%")
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d1117")
                        ax.barh(class_names, probs,
                                color=["#3fb950" if i == true_class_idx else "#58a6ff" if i == pred_class else "#30363d"
                                       for i in range(4)])
                        ax.set_title("Class Probabilities", color="white")
                        ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
                        for spine in ax.spines.values(): spine.set_color("#30363d")
                        st.pyplot(fig); plt.close()
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Architecture Viz":
# ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("## 📐 LSTM Architecture Visualization")
    st.markdown("Visual breakdown of the LSTM memory cell from Figure 1 of the paper.")

    tab1, tab2, tab3 = st.tabs(["🔬 Activation Functions", "📉 Vanishing Gradient", "🔁 CEC Explained"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
        x = np.linspace(-6, 6, 400)
        f_sig = 1 / (1 + np.exp(-x))
        h_func = 2 / (1 + np.exp(-x)) - 1
        g_func = 4 / (1 + np.exp(-x)) - 2

        axes[0].plot(x, f_sig, color="#58a6ff", lw=2.5, label="f(x) = 1/(1+e^-x)  [gates]")
        axes[0].plot(x, h_func, color="#3fb950", lw=2.5, label="h(x) = 2/(1+e^-x)-1  [cell out]")
        axes[0].plot(x, g_func, color="#f85149", lw=2.5, label="g(x) = 4/(1+e^-x)-2  [cell in]")
        axes[0].axhline(0, color="#8b949e", lw=0.8, ls="--")
        axes[0].axvline(0, color="#8b949e", lw=0.8, ls="--")
        axes[0].set_title("Paper Activation Functions (Eq. 3,4,5)", color="white", fontsize=11)

        # Derivative plot
        df = f_sig * (1 - f_sig)
        dh = 2 * f_sig * (1 - f_sig)
        axes[1].plot(x, df, color="#58a6ff", lw=2.5, label="f'(x) — max 0.25")
        axes[1].plot(x, dh, color="#3fb950", lw=2.5, label="h'(x) — max 0.5")
        axes[1].axhline(0.25, color="#58a6ff", lw=1, ls=":", alpha=0.5)
        axes[1].set_title("Derivatives (key to vanishing gradient)", color="white", fontsize=11)
        axes[1].set_ylabel("Derivative Value", color="white")

        for ax in axes:
            ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
            ax.grid(True, alpha=0.2, color="white"); ax.legend(facecolor="#161b22", labelcolor="white")
            for spine in ax.spines.values(): spine.set_color("#30363d")
            ax.set_xlabel("x", color="white")

        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
        q = np.arange(1, 101)
        for w, c, label in [(0.9, "#f85149", "RNN: |w·f'|=0.9"),
                             (0.7, "#d29922", "RNN: |w·f'|=0.7"),
                             (0.5, "#8b949e", "RNN: |w·f'|=0.5")]:
            ax.semilogy(q, w**q, color=c, lw=2.5, label=label)
        ax.axhline(1.0, color="#3fb950", lw=3, label="LSTM CEC (constant = 1.0)", ls="-")
        ax.fill_between(q, 1e-10, 1e-2, alpha=0.1, color="#f85149", label="Vanishing zone")
        ax.set_title("Vanishing Gradient: RNN vs LSTM CEC (Paper Eq.1,2,35)", color="white", fontsize=13)
        ax.set_xlabel("Time Steps Backward (q)", color="white")
        ax.set_ylabel("Gradient Magnitude (log)", color="white")
        ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white"); ax.legend(facecolor="#161b22", labelcolor="white")
        for spine in ax.spines.values(): spine.set_color("#30363d")
        ax.set_xlim(1, 100); ax.set_ylim(1e-10, 10)
        st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### Constant Error Carousel (CEC) — Core Innovation")
        st.latex(r"s_c(t) = s_c(t-1) + y^{in}(t) \cdot g(net_c(t))")
        st.latex(r"\frac{\partial s_c(t-k)}{\partial s_c(t-k-1)} \approx 1 \quad \text{(Eq. 35)}")
        st.success("The internal state flows **without scaling** — error can travel back 1000+ steps!")

        st.markdown("#### Gate Roles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔵 Input Gate** `y^{in}(t)`
            - Controls **write** access to CEC
            - Prevents input weight conflicts
            - Learns *when* to update memory
            - Initially biased negative (−1, −3, −5...)
            """)
        with col2:
            st.markdown("""
            **🔴 Output Gate** `y^{out}(t)`
            - Controls **read** access from CEC
            - Prevents output weight conflicts
            - Learns *when* to reveal memory
            - Initially biased negative (−2, −4, −6...)
            """)

        st.markdown("#### Abuse Problem & Solutions (Section 4)")
        st.info("""
        **Problem**: Early training may use memory cells as bias cells (constant activation).
        **Solutions**:
        1. Sequential network construction — add cells when error stops decreasing
        2. **Output gate negative bias** — ensures cells start near zero, get allocated later
        """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:0.8rem; font-family:IBM Plex Mono, monospace'>
    📄 Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. 
    Neural Computation, 9(8), 1735–1780. &nbsp;|&nbsp; GUI built with Streamlit
</div>
""", unsafe_allow_html=True)
