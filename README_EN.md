# EcoCondomínio Pro

**EcoCondomínio Pro** is a data-driven platform to **measure, audit, and optimize** waste management in condominiums. It turns weekly collection entries into **operational insights, environmental impact, and savings**, with **executive dashboards** and **premium reports (PDF/Excel)** ready for presentations.

> **Stack**: Python · Streamlit · Plotly · Pandas · SQLite · FPDF (fpdf2) · XlsxWriter · Kaleido

---

## Features

- **Smart data entry**
  - Real-time validation (apartment format, limits, weights).
  - Advanced filters by **apartment / block / week / period**.
- **Executive Dashboard**
  - **Sparklines** (Total, Recyclable, CO₂, Adherence).
  - Weekly evolution (stacked area + **moving average**).
  - Total × CO₂ (secondary axis), 100% composition per week.
  - Treemap Block→Apartment, Pareto, Control Chart (X-bar), Heatmap, Bubble scatter.
- **Enterprise Reports**
  - **Premium PDF**: cover, executive cards, chart pages, zebra ranking table, insights, paginated footer (Unicode-ready).
  - **Advanced Excel**: dashboard, apartment summary, raw data, time series analysis, rankings (with conditional formatting).
- **Robust architecture**
  - SQLite with indexes, constraints, and **audit triggers** (INSERT/UPDATE).
  - **Automatic backups**, **smart cache** (TTL), and modern CSS.

---

## Architecture

```
ecocondominio-pro/
├─ ecocondominio_pro.py           # Streamlit app
├─ data/
│  ├─ ecocondominio.db            # SQLite DB (auto-created)
│  ├─ backups/                    # Automatic backups (rotation)
│  └─ exports/                    # Exported files (if applicable)
├─ assets/
│  ├─ DejaVuSans.ttf              # (optional) Unicode font for PDF
│  └─ DejaVuSans-Bold.ttf         # (optional) Unicode font for PDF
└─ app.log                        # Application log
```

SQLite tables: `measurements`, `settings`, `audit_log` (with indexes & triggers).

---

## Installation

> Requires **Python 3.10+**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

If you prefer without `requirements.txt`:

```bash
pip install streamlit pandas numpy plotly fpdf2 xlsxwriter kaleido pillow
```

---

## Run

```bash
streamlit run ecocondominio_pro.py
```

> The app opens in your browser (see the terminal for the URL).

---

## Export (PDF / Excel)

Open **“Enterprise Reports”** inside the app:
- **Generate PDF**: premium executive report.
- **Generate Excel**: multi-sheet workbook with summary, raw data, time analysis, and rankings.

**Unicode fonts (optional)** for perfect accents/symbols in the PDF: place `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` under `assets/`. Without them, the app falls back to ASCII-safe mode to avoid crashes.

---

## Troubleshooting

- **Charts don’t appear in PDF** → `pip install kaleido`
- **Invalid character in PDF (“•”, “—”)** → add Unicode fonts (above) or rely on ASCII fallback already implemented.
- **“Not enough horizontal space…”** → already handled in layout (cursor reset before `cell(0, …)` and `multi_cell(0, …)`). If you add very long custom titles/notes, consider shortening them.

---

## Publish to GitHub (quick)

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
