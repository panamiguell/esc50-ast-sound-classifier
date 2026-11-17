"""Regenera la Fig. 2 del informe (F1 de las clases más difíciles y confusión por categoría)
a partir de las salidas guardadas por notebooks/02_solucion_ast_esc50_v2.ipynb, con letra más grande.

Uso: python scripts/figura_resultados_informe.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "v2" / "tables"
CATS = ["Animales", "Naturales y agua", "Humanos no verbales", "Domésticos", "Urbanos"]

split = pd.read_csv(ROOT / "data" / "splits" / "split.csv")
test = split[split.split == "test"].sort_values("idx")
y = test.label.to_numpy()
P = np.load(TAB / "ensamble_prob_test.npy")
pred = P.argmax(1)
pc = pd.read_csv(TAB / "metricas_por_clase.csv", index_col=0)

# Comprobación: las predicciones reproducen las métricas guardadas del ensamble
print(f"exactitud {accuracy_score(y, pred):.4f} | F1-macro {f1_score(y, pred, average='macro'):.4f}")
cls_cat = dict(zip(test.label, test.categoria))
yc = np.array([CATS.index(cls_cat[i]) for i in y])
pcat = np.array([CATS.index(cls_cat[i]) for i in pred])
cmc = confusion_matrix(yc, pcat, labels=range(5))
print(cmc)

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
peores = pc.sort_values(["f1", "recall"]).head(12).iloc[::-1]
cols = {cat: plt.cm.tab10(k) for k, cat in enumerate(CATS)}
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw={"width_ratios": [1.05, 1.1], "wspace": 0.72})
ax[0].barh(range(len(peores)), peores.f1.values, color=[cols[c] for c in peores.categoria])
ax[0].set_yticks(range(len(peores)), [c.replace("_", " ") for c in peores.index], fontsize=7.5)
for t, c in zip(ax[0].get_yticklabels(), peores.categoria):
    t.set_color(cols[c])
ax[0].set_xlim(0.4, 1.0); ax[0].set_xticks([0.4, 0.6, 0.8, 1.0], ["0,4", "0,6", "0,8", "1,0"])
ax[0].set_xlabel("F1 (prueba)"); ax[0].set_title("(a) 12 clases con menor F1", fontsize=8)
ax[1].imshow(cmc / cmc.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax[1].text(j, i, cmc[i, j], ha="center", va="center", fontsize=8,
                   color="white" if cmc[i, j] / cmc[i].sum() > 0.5 else "black")
ab = ["Anim.", "Nat.", "Hum.", "Dom.", "Urb."]
ax[1].set_xticks(range(5), ab, rotation=90, fontsize=7.5); ax[1].set_yticks(range(5), ab, fontsize=7.5)
ax[1].set_xlabel("Predicción"); ax[1].set_ylabel("Real"); ax[1].set_title("(b) Confusión por categoría", fontsize=8)
out = ROOT / "report" / "figs" / "v2_fig1_resultados.pdf"
plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("Figura guardada en", out.relative_to(ROOT))
