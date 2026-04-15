# 🧠 LSTM Paper Implementation
## Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
### Neural Computation 9(8):1735–1780

---

## 📂 Project Structure

```
lstm_project/
├── LSTM_Paper_Implementation.ipynb   # Google Colab notebook (all experiments)
├── app.py                            # Streamlit GUI
├── requirements.txt                  # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Google Colab Notebook
Open `LSTM_Paper_Implementation.ipynb` in Google Colab:
- [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
- Upload the notebook or paste the code
- Runtime → Change runtime type → **GPU**
- Run All Cells

### 2. Streamlit GUI (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Streamlit GUI (Colab + ngrok)
```python
!pip install streamlit pyngrok -q
!streamlit run app.py &
from pyngrok import ngrok
url = ngrok.connect(8501)
print(url)
```

---

## 🧪 Experiments Implemented

| Experiment | Paper Section | Description |
|:---:|:---:|:---|
| Exp 2a | §5.2 | Noise-free Long Time Lag (p=10,100) |
| Exp 4 | §5.4 | Adding Problem (T=100, min lag=50) |
| Exp 6a | §5.6 | Temporal Order (2 widely separated symbols) |
| Architecture | §3,4 | CEC, Activation Functions, Vanishing Gradient |

---

## 🔑 Key Concepts from Paper

### Constant Error Carousel (CEC)
```
s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t))
∂s_c(t-k)/∂s_c(t-k-1) ≈ 1  [Eq. 35]
```
Error flows **without scaling** — bridges 1000+ time steps!

### Activation Functions
- `f(x) = 1/(1+exp(-x))` — gates, range [0,1] — Eq.(3)
- `h(x) = 2/(1+exp(-x)) - 1` — cell output, range [-1,1] — Eq.(4)  
- `g(x) = 4/(1+exp(-x)) - 2` — cell input, range [-2,2] — Eq.(5)

### Gate Roles
- **Input gate**: Controls write access — prevents input weight conflicts
- **Output gate**: Controls read access — prevents output weight conflicts

---

## 📊 Models Compared

| Model | Long Time Lag | Gradient Flow | Paper Result |
|:---:|:---:|:---:|:---:|
| **LSTM** | ✅ Solves | Constant (CEC) | **100% success** |
| Vanilla RNN | ❌ Fails | Vanishes exponentially | 0% (p≥10) |
| GRU | ⚠️ Partial | Reset gate | Post-1997 |
| BiLSTM | ✅ Solves | Bidirectional CEC | Better context |

---

## 📖 Citation

```bibtex
@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}
```
