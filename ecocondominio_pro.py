#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoCondomínio Pro - Streamlit App (versão 3.1.1, revisada)
- Correção: adicionadas chaves únicas (key=...) a todos os widgets para evitar
  "multiple selectbox elements with the same auto-generated ID".
"""

import os
import io
import base64
import json
import sqlite3
import tempfile
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from fpdf import FPDF

# ==============================
# LOGGING AVANÇADO
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EcoCondominioPro")


# ==============================
# CONFIGURAÇÕES E CONSTANTES
# ==============================
@dataclass
class EnvironmentalConstants:
    CO2_EMISSIONS = {"reciclavel": 0.15, "organico": 0.45, "rejeito": 1.80}
    DISPOSAL_COSTS = {"reciclavel": 0.08, "organico": 0.12, "rejeito": 0.45}
    RECYCLING_REVENUE = {"reciclavel": 2.30, "organico": 0.15, "rejeito": 0.00}
    WATER_SAVINGS = {"reciclavel": 15.5, "organico": 2.1, "rejeito": 0.0}


@dataclass
class AppConfig:
    PAGE_TITLE = "EcoCondomínio Pro"
    PAGE_ICON = "🌿"
    LAYOUT = "wide"

    BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    EXPORTS_DIR = DATA_DIR / "exports"
    DB_NAME = "ecocondominio.db"
    BACKUP_INTERVAL_DAYS = 7

    THEME_COLORS = {
        "primary":   "#059669",
        "secondary": "#0ea5e9",
        "success":   "#10b981",
        "warning":   "#f59e0b",
        "danger":    "#ef4444",
        "info":      "#3b82f6",
        "dark":      "#1f2937",
        "light":     "#f8fafc",
    }

    MAX_WEIGHT_KG = 1000.0
    MAX_APARTMENTS = 50


class SmartCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, Any] = {}
        self._times: Dict[str, float] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        now = datetime.now().timestamp()
        if key in self._cache and (now - self._times.get(key, 0)) < self.ttl:
            return self._cache[key]
        if key in self._cache:
            self._cache.pop(key, None)
            self._times.pop(key, None)
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._times[key] = datetime.now().timestamp()

    def clear(self) -> None:
        self._cache.clear()
        self._times.clear()


cache = SmartCache()


def get_modern_css() -> str:
    c = AppConfig.THEME_COLORS
    return f"""
    <style>
      :root {{
        --primary: {c['primary']}; --secondary: {c['secondary']}; --success:{c['success']};
        --warning:{c['warning']}; --danger:{c['danger']}; --dark:{c['dark']}; --light:{c['light']};
        --gradient-primary: linear-gradient(135deg, var(--primary), var(--secondary));
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,.1);
        --shadow-xl: 0 20px 25px -5px rgba(0,0,0,.1);
      }}
      .main .block-container {{ padding: 2rem 1rem; max-width: 1400px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 20px; box-shadow: var(--shadow-xl); }}
      .app-header {{ background: var(--gradient-primary); color: #fff; padding: 2rem; border-radius: 16px; text-align: center; }}
      .app-title {{ font-size: 2.5rem; font-weight: 800; margin: 0; }}
      .stButton > button {{ background: var(--gradient-primary) !important; color:#fff !important; border:none !important; border-radius:12px !important; padding:.75rem 2rem !important; }}
    </style>
    """


class Utils:
    @staticmethod
    def format_currency(value: float, currency: str = "R$") -> str:
        try:
            return f"{currency} " + f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return f"{currency} 0,00"

    @staticmethod
    def format_weight(value: float, unit: str = "kg") -> str:
        try:
            return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + f" {unit}"
        except Exception:
            return f"0,00 {unit}"

    @staticmethod
    def calculate_sustainability_score(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        total = df[["recyclable_kg", "organic_kg", "waste_kg"]].sum().sum()
        if total <= 0:
            return 0.0
        r = df["recyclable_kg"].sum() / total
        o = df["organic_kg"].sum() / total
        w = df["waste_kg"].sum() / total
        return float(np.clip(r*100 + o*80 + w*20, 0, 100))

    @staticmethod
    def validate_apartment_format(apt: str) -> bool:
        import re
        pattern = r'^(Apto\s+)?\d{2,4}[A-Z]?$|^[A-Z]\d{2,4}$'
        return bool(re.match(pattern, apt.strip(), re.IGNORECASE))


class BackupManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.backup_dir = self.db_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> bool:
        try:
            import shutil
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self.backup_dir / f"backup_{ts}.db"
            shutil.copy2(self.db_path, dest)
            backups = sorted(self.backup_dir.glob("backup_*.db"))
            for old in backups[:-10]:
                old.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def should_backup(self) -> bool:
        backups = list(self.backup_dir.glob("backup_*.db"))
        if not backups:
            return True
        latest = max(backups, key=lambda p: p.stat().st_mtime)
        days = (datetime.now().timestamp() - latest.stat().st_mtime) / 86400
        return days >= AppConfig.BACKUP_INTERVAL_DAYS


class AdvancedDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup = BackupManager(db_path)
        self._init_database()
        self._check_backup()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_database(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week INTEGER NOT NULL CHECK(week BETWEEN 1 AND 52),
                    block TEXT NOT NULL,
                    apartment TEXT NOT NULL,
                    recyclable_kg REAL NOT NULL DEFAULT 0 CHECK(recyclable_kg >= 0),
                    organic_kg REAL NOT NULL DEFAULT 0 CHECK(organic_kg >= 0),
                    waste_kg REAL NOT NULL DEFAULT 0 CHECK(waste_kg >= 0),
                    participating_apts INTEGER NOT NULL DEFAULT 0 CHECK(participating_apts >= 0),
                    reference_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    UNIQUE(week, block, apartment, reference_date)
                );
            """)
            cur.executescript("""
                CREATE INDEX IF NOT EXISTS idx_m_week ON measurements(week);
                CREATE INDEX IF NOT EXISTS idx_m_block ON measurements(block);
                CREATE INDEX IF NOT EXISTS idx_m_apt ON measurements(apartment);
                CREATE INDEX IF NOT EXISTS idx_m_date ON measurements(reference_date);
            """)
            conn.commit()

    def _check_backup(self) -> None:
        if self.backup.should_backup():
            self.backup.create_backup()

    def save_measurement(self, data: Dict) -> bool:
        try:
            if not Utils.validate_apartment_format(data["apartment"]):
                raise ValueError("Apartamento inválido.")
            total = float(data["recyclable_kg"]) + float(data["organic_kg"]) + float(data["waste_kg"])
            if total > AppConfig.MAX_WEIGHT_KG:
                raise ValueError("Peso total acima do limite.")
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO measurements
                    (week, block, apartment, recyclable_kg, organic_kg, waste_kg,
                     participating_apts, reference_date, notes)
                    VALUES (?,?,?,?,?,?,?,?,?);
                """, (
                    int(data["week"]), str(data["block"]).strip(), str(data["apartment"]).strip(),
                    float(data["recyclable_kg"]), float(data["organic_kg"]), float(data["waste_kg"]),
                    int(data["participating_apts"]), str(data["reference_date"]), str(data.get("notes","")).strip()
                ))
                conn.commit()
            cache.clear()
            return True
        except Exception as e:
            logger.error(f"save_measurement: {e}")
            return False

    def load_measurements(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        try:
            query = "SELECT id, week, block, apartment, recyclable_kg, organic_kg, waste_kg, participating_apts, reference_date, created_at, updated_at, notes FROM measurements"
            clauses = []
            params: List[Any] = []
            if filters:
                if "apartment" in filters:
                    clauses.append("apartment = ?")
                    params.append(filters["apartment"])
                if "block" in filters:
                    clauses.append("block = ?")
                    params.append(filters["block"])
                if "week_range" in filters:
                    clauses.append("week BETWEEN ? AND ?")
                    params.extend([int(filters["week_range"][0]), int(filters["week_range"][1])])
                if "date_range" in filters and isinstance(filters["date_range"], (list, tuple)) and len(filters["date_range"]) == 2:
                    clauses.append("reference_date BETWEEN ? AND ?")
                    params.extend([str(filters["date_range"][0]), str(filters["date_range"][1])])
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY reference_date DESC, block ASC, apartment ASC"

            with self._connect() as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=["reference_date", "created_at", "updated_at"])

            if df.empty:
                return df
            env = EnvironmentalConstants()
            df["total_kg"] = df[["recyclable_kg", "organic_kg", "waste_kg"]].sum(axis=1)
            df["co2_emissions_kg"] = (
                df["recyclable_kg"] * env.CO2_EMISSIONS["reciclavel"] +
                df["organic_kg"]   * env.CO2_EMISSIONS["organico"] +
                df["waste_kg"]     * env.CO2_EMISSIONS["rejeito"]
            ).astype(float)
            df["disposal_cost_brl"] = (
                df["recyclable_kg"] * env.DISPOSAL_COSTS["reciclavel"] +
                df["organic_kg"]   * env.DISPOSAL_COSTS["organico"] +
                df["waste_kg"]     * env.DISPOSAL_COSTS["rejeito"]
            ).astype(float)
            df["recycling_revenue_brl"] = (
                df["recyclable_kg"] * env.RECYCLING_REVENUE["reciclavel"] +
                df["organic_kg"]   * env.RECYCLING_REVENUE["organico"]
            ).astype(float)
            df["water_savings_liters"] = (
                df["recyclable_kg"] * env.WATER_SAVINGS["reciclavel"] +
                df["organic_kg"]   * env.WATER_SAVINGS["organico"]
            ).astype(float)
            return df
        except Exception as e:
            logger.error(f"load_measurements: {e}")
            return pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        ck = "stats"
        c = cache.get(ck)
        if c is not None:
            return c
        stats: Dict[str, Any] = {}
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM measurements;")
                stats["total_records"] = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT apartment) FROM measurements;")
                stats["unique_apartments"] = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT week) FROM measurements;")
                stats["covered_weeks"] = cur.fetchone()[0]
                cur.execute("""
                    SELECT COALESCE(SUM(recyclable_kg),0),
                           COALESCE(SUM(organic_kg),0),
                           COALESCE(SUM(waste_kg),0),
                           COALESCE(SUM(participating_apts),0)
                    FROM measurements;
                """)
                r = cur.fetchone() or (0,0,0,0)
                stats["total_recyclable_kg"] = float(r[0]); stats["total_organic_kg"] = float(r[1]); stats["total_waste_kg"] = float(r[2]); stats["total_participation"] = int(r[3])
                cur.execute("""
                    SELECT AVG(recyclable_kg + organic_kg + waste_kg),
                           AVG(participating_apts) FROM measurements;
                """)
                r = cur.fetchone() or (0,0)
                stats["avg_total_per_record"] = float(r[0] or 0); stats["avg_participation"] = float(r[1] or 0)
        except Exception as e:
            logger.error(f"get_statistics: {e}")
        cache.set(ck, stats)
        return stats


class EnterpriseReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_comprehensive_summary(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        g = self.df.groupby("apartment", as_index=False).agg(
            recyclable_kg_sum=("recyclable_kg","sum"),
            organic_kg_sum=("organic_kg","sum"),
            waste_kg_sum=("waste_kg","sum"),
            total_kg_sum=("total_kg","sum"),
            total_kg_mean=("total_kg","mean"),
            total_kg_max=("total_kg","max"),
            total_kg_min=("total_kg","min"),
            participating_sum=("participating_apts","sum"),
            recycling_revenue_brl_sum=("recycling_revenue_brl","sum"),
            co2_emissions_kg_sum=("co2_emissions_kg","sum"),
            water_savings_liters_sum=("water_savings_liters","sum"),
            count=("id","count")
        )
        def _score(apt: str) -> float:
            sub = self.df[self.df["apartment"]==apt][["recyclable_kg","organic_kg","waste_kg"]]
            return Utils.calculate_sustainability_score(sub)
        g["sustainability_score"] = g["apartment"].apply(_score).round(1)
        denom = (g["recyclable_kg_sum"] + g["organic_kg_sum"] + g["waste_kg_sum"]).replace(0, np.nan)
        g["recycling_rate_percent"] = (g["recyclable_kg_sum"]/denom*100).fillna(0).round(1)
        g["sustainability_rank"] = g["sustainability_score"].rank(method="dense", ascending=False).astype(int)
        return g.sort_values(["sustainability_score","recycling_rate_percent"], ascending=False)

    def generate_advanced_excel(self) -> bytes:
        from pandas import ExcelWriter
        import xlsxwriter
        summary = self.calculate_comprehensive_summary()
        buf = io.BytesIO()
        with ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb = writer.book
            header_fmt = wb.add_format({"bold":True,"fg_color":"#059669","font_color":"white","border":1})
            # Dashboard
            dash = pd.DataFrame({
                "Métrica":[
                    "Total de Apartamentos","Total de Resíduos (kg)","Total Reciclável (kg)",
                    "Total Orgânico (kg)","Total Rejeito (kg)","Emissões CO₂ (kg)",
                    "Economia com Reciclagem (R$)","Economia de Água (L)","Score Médio de Sustentabilidade"
                ],
                "Valor":[
                    self.df["apartment"].nunique(), self.df["total_kg"].sum(), self.df["recyclable_kg"].sum(),
                    self.df["organic_kg"].sum(), self.df["waste_kg"].sum(), self.df["co2_emissions_kg"].sum(),
                    self.df["recycling_revenue_brl"].sum(), self.df["water_savings_liters"].sum(),
                    Utils.calculate_sustainability_score(self.df)
                ]
            })
            dash.to_excel(writer, sheet_name="Dashboard Executivo", index=False)
            ws = writer.sheets["Dashboard Executivo"]
            for col, name in enumerate(dash.columns):
                ws.write(0, col, name, header_fmt)
            ws.set_column("A:A", 36); ws.set_column("B:B", 18)

            # Resumo por Apartamento
            if not summary.empty:
                summary.to_excel(writer, sheet_name="Resumo por Apartamento", index=False)
                ws = writer.sheets["Resumo por Apartamento"]
                for col, name in enumerate(summary.columns):
                    ws.write(0, col, name, header_fmt)
                ws.set_column(0, len(summary.columns)-1, 16)

            # Dados Detalhados
            self.df.to_excel(writer, sheet_name="Dados Detalhados", index=False)
            ws = writer.sheets["Dados Detalhados"]
            for col, name in enumerate(self.df.columns):
                ws.write(0, col, name, header_fmt)
            ws.set_column(0, len(self.df.columns)-1, 16)
        return buf.getvalue()

    def _create_pdf_charts(self, tmp_dir: Path) -> List[Path]:
        paths: List[Path] = []
        try:
            temporal = self.df.groupby("week", as_index=False).agg(
                recyclable_kg=("recyclable_kg","sum"),
                organic_kg=("organic_kg","sum"),
                waste_kg=("waste_kg","sum"),
            )
            fig1 = px.line(temporal, x="week", y=["recyclable_kg","organic_kg","waste_kg"],
                           title="Evolução dos Resíduos por Semana", markers=True, template="plotly_white")
            p1 = tmp_dir / "evolucao.png"
            try:
                fig1.write_image(str(p1), width=1000, height=450, scale=2)
                paths.append(p1)
            except Exception:
                pass
        except Exception:
            pass
        return paths

    def generate_professional_pdf(self) -> bytes:
        """Gera PDF premium (capa, cards executivos, gráficos, ranking, rodapé com paginação).
        - Usa fonte Unicode (DejaVu) se disponível em assets/, senão fallback para Arial.
        - Aproveita melhor os espaços com layout em sessões.
        """
        # ---- Preparação de dados ----
        summary = self.calculate_comprehensive_summary()
        total_regs = len(self.df)
        total_apts = self.df["apartment"].nunique()
        total_kg = float(self.df.get("total_kg", self.df[["recyclable_kg","organic_kg","waste_kg"]].sum(axis=1)).sum())
        total_rec = float(self.df["recyclable_kg"].sum())
        total_org = float(self.df["organic_kg"].sum())
        total_waste = float(self.df["waste_kg"].sum())
        total_co2 = float(self.df.get("co2_emissions_kg", 0).sum())
        total_water = float(self.df.get("water_savings_liters", 0).sum())
        total_revenue = float(self.df.get("recycling_revenue_brl", 0).sum())
        score = Utils.calculate_sustainability_score(self.df)

        # ---- Classe PDF com rodapé padrão ----
        class PDF(FPDF):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.footer_text_left = "Relatorio gerado automaticamente pelo EcoCondominio Pro"
            def footer(self):
                self.set_y(-12)
                self.set_text_color(120,120,120)
                # esquerda
                self.set_font('Arial', '', 8)
                self.set_x(self.l_margin)
                self.cell(0, 6, self.footer_text_left, 0, 0, 'L')
                # direita
                self.set_y(-12)
                self.set_x(self.w - self.r_margin - 20)
                self.cell(20, 6, f"{self.page_no()}/{{nb}}", 0, 0, 'R')

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()

        # ---- Fontes ----
        base_dir = AppConfig.BASE_DIR
        assets_dir = base_dir / "assets"
        unicode_font_used = False
        try:
            dejavu = assets_dir / "DejaVuSans.ttf"
            dejavu_bold = assets_dir / "DejaVuSans-Bold.ttf"
            if dejavu.exists():
                pdf.add_font("DejaVu","",str(dejavu),uni=True)
                unicode_font_used = True
            if dejavu_bold.exists():
                pdf.add_font("DejaVu","B",str(dejavu_bold),uni=True)
        except Exception:
            unicode_font_used = False

        
        # Escolha de marcador (bullet) compatível
        bullet = '• ' if unicode_font_used else '- '
        def set_h1():
            if unicode_font_used: pdf.set_font("DejaVu","B",20)
            else: pdf.set_font("Arial","B",20)
        def set_h2():
            if unicode_font_used: pdf.set_font("DejaVu","B",14)
            else: pdf.set_font("Arial","B",14)
        def set_body():
            if unicode_font_used: pdf.set_font("DejaVu","",11)
            else: pdf.set_font("Arial","",11)
        def set_small():
            if unicode_font_used: pdf.set_font("DejaVu","",9)
            else: pdf.set_font("Arial","",9)

        # ---- Funções de desenho ----
        def draw_section_title(text):
            set_h2(); pdf.set_text_color(31,41,55); pdf.cell(0,10,text,ln=True)

        def draw_card(x, y, w, h, title, value, accent=(5,150,105)):
            # contêiner
            pdf.set_xy(x,y)
            pdf.set_fill_color(248,250,252)
            pdf.set_draw_color(230,235,240)
            pdf.rect(x,y,w,h,'F')
            pdf.rect(x,y,w,h,'D')
            # título
            set_small(); pdf.set_text_color(90,98,110); pdf.set_xy(x+4,y+5)
            pdf.cell(w-8,5,title,0,1,'L')
            # valor
            set_h2(); pdf.set_text_color(*accent); pdf.set_xy(x+4,y+12)
            pdf.cell(w-8,8,value,0,0,'L')
            pdf.set_text_color(0,0,0)

        # ---- CAPA ----
        pdf.add_page()
        # "gradiente" simples com faixas
        for i, g in enumerate(range(0, 255, 6)):
            pdf.set_fill_color(240-g//6, 255-g//5, 248)
            pdf.rect(0, i*2, pdf.w, 2, 'F')

        set_h1(); pdf.set_text_color(5,150,105)
        pdf.ln(40)
        pdf.set_x(pdf.l_margin)

        pdf.cell(0,12,"EcoCondominio Pro - Relatorio Executivo",ln=True,align="C")
        set_body(); pdf.set_text_color(33,37,41)
        pdf.set_x(pdf.l_margin)

        pdf.cell(0,8,f"Gerado em {datetime.now().strftime('%d/%m/%Y as %H:%M')}",ln=True,align="C")
        pdf.ln(10)
        set_small(); pdf.set_text_color(120,120,120)
        pdf.set_x(pdf.l_margin)

        pdf.cell(0,7,f"Base: {total_regs} registros | Apartamentos: {total_apts}",ln=True,align="C")
        pdf.set_text_color(0,0,0)

        # ---- RESUMO EXECUTIVO (cards 2x2) ----
        pdf.add_page()
        draw_section_title("Resumo Executivo")
        card_w, card_h = (pdf.w-30)/2, 24
        x0, y0 = 15, pdf.get_y()+2
        draw_card(x0, y0, card_w, card_h, "Total de Residuos (kg)", f"{total_kg:,.1f}".replace(',','X').replace('.',',').replace('X','.' ))
        draw_card(x0+card_w+5, y0, card_w, card_h, "Score de Sustentabilidade", f"{score:.1f}/100")
        draw_card(x0, y0+card_h+8, card_w, card_h, "Receita com Reciclagem", Utils.format_currency(total_revenue))
        draw_card(x0+card_w+5, y0+card_h+8, card_w, card_h, "Agua Poupada (L)", f"{total_water:,.0f}".replace(',','X').replace('.',',').replace('X','.'))
        pdf.ln(card_h*2+18)

        # ---- KPIs adicionais em texto curto ----
        set_body()
        bullets = [
            f"Reciclavel: {total_rec:.1f} kg",
            f"Organico: {total_org:.1f} kg",
            f"Rejeito: {total_waste:.1f} kg",
            f"Emissoes CO2: {total_co2:.1f} kg"
        ]
        pdf.set_text_color(55,65,81)
        for b in bullets:
            pdf.set_x(pdf.l_margin)

            pdf.cell(0,7, bullet + b, ln=True)
        pdf.set_text_color(0,0,0)

        # ---- GRAFICOS ----
        with tempfile.TemporaryDirectory() as tmp_dir:
            chart_paths = self._create_pdf_charts(Path(tmp_dir))

            # Evolução temporal
            pdf.add_page()
            draw_section_title("Evolucao Temporal")
            if chart_paths and Path(chart_paths[0]).exists():
                pdf.image(str(chart_paths[0]), x=12, w=pdf.w-24)
                pdf.set_x(pdf.l_margin)
            else:
                set_body(); pdf.multi_cell(0,7,"[Grafico nao disponivel - instale 'kaleido' para exportar imagens Plotly]")

            # Top 10
            pdf.add_page()
            draw_section_title("Top 10 Apartamentos - Volume Total")
            if len(chart_paths) > 1 and Path(chart_paths[1]).exists():
                pdf.image(str(chart_paths[1]), x=12, w=pdf.w-24)
                pdf.set_x(pdf.l_margin)
            else:
                set_body(); pdf.multi_cell(0,7,"[Grafico nao disponivel]")

            # Distribuicao tipos
            pdf.add_page()
            draw_section_title("Distribuicao por Tipo de Residuo")
            if len(chart_paths) > 2 and Path(chart_paths[2]).exists():
                pdf.image(str(chart_paths[2]), x=12, w=pdf.w-24)
                pdf.set_x(pdf.l_margin)
            else:
                set_body(); pdf.multi_cell(0,7,"[Grafico nao disponivel]")

        # ---- RANKING (tabela zebrada) ----
        pdf.add_page()
        draw_section_title("Ranking - Resumo por Apartamento (Top 15)")
        set_small()
        headers = ["Apto","Recic.(kg)","Org.(kg)","Rej.(kg)","Score","Rank"]
        widths = [30,28,28,28,28,20]
        # cabeçalho
        pdf.set_fill_color(5,150,105); pdf.set_text_color(255,255,255)
        for h, w in zip(headers, widths):
            pdf.cell(w,8,h,1,0,'C',True)
        pdf.ln(8); pdf.set_text_color(0,0,0)

        # linhas
        if summary.empty:
            set_body(); pdf.multi_cell(0,7,"Sem dados para ranking.")
        else:
            set_small()
            top = summary.head(15)
            fill = False
            for _, row in top.iterrows():
                pdf.set_fill_color(248,250,252) if fill else pdf.set_fill_color(255,255,255)
                vals = [
                    str(row['apartment'])[:12],
                    f"{row['recyclable_kg_sum']:.1f}",
                    f"{row['organic_kg_sum']:.1f}",
                    f"{row['waste_kg_sum']:.1f}",
                    f"{row['sustainability_score']:.1f}",
                    str(int(row['sustainability_rank']))
                ]
                for v, w in zip(vals, widths):
                    pdf.cell(w,7,v,1,0,'C',True)
                pdf.ln(7)
                fill = not fill

        # ---- Insights finais ----
        pdf.add_page()
        draw_section_title("Conclusoes & Insights")
        set_body()
        insights = []
        # Semana mais produtiva
        if not self.df.empty:
            by_week = self.df.groupby("week")["total_kg"].sum()
            if not by_week.empty:
                wk = int(by_week.idxmax()); val = float(by_week.max())
                insights.append(f"Semana {wk} apresentou o maior volume: {val:.1f} kg.")
        # Apto destaque
        if not summary.empty:
            best = summary.sort_values("sustainability_score", ascending=False).iloc[0]
            insights.append(f"Apartamento destaque: {best['apartment']} (score {best['sustainability_score']:.1f}).")
        # Potencial de melhoria
        recyclable_potential = total_waste * 0.3
        if recyclable_potential > 0:
            potential_revenue = recyclable_potential * EnvironmentalConstants.RECYCLING_REVENUE["reciclavel"]
            insights.append(f"Potencial: {recyclable_potential:.1f} kg de rejeito poderiam virar reciclagem, gerando {Utils.format_currency(potential_revenue)}.")

        if not insights:
            insights = ["Sem insights adicionais disponiveis para o periodo selecionado."]
        for txt in insights:
            pdf.set_x(pdf.l_margin)

            pdf.multi_cell(0,7, bullet + txt)

        # ---- Saida bytes robusta ----
        out = pdf.output(dest="S")
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        return out.encode("latin-1")

def render_app_header():
    st.markdown("""
    <div class="app-header">
    <h1 class="app-title">🌿 EcoCondomínio Pro</h1>
    <p class="app-subtitle">Sistema Inteligente de Gestão de Resíduos Sustentáveis</p>
    </div>
    """, unsafe_allow_html=True)
    

def render_quick_stats(stats: Dict[str, Any]):
    if not stats: return
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("📊 Total de Registros", f"{stats.get('total_records',0):,}", delta=f"{stats.get('covered_weeks',0)} semanas")
    with c2:
        st.metric("🏠 Apartamentos Ativos", stats.get("unique_apartments",0), delta=f"Média {stats.get('avg_participation',0):.1f} aderentes")
    with c3:
        total_kg = stats.get("total_recyclable_kg",0) + stats.get("total_organic_kg",0) + stats.get("total_waste_kg",0)
        st.metric("⚖️ Total Processado", Utils.format_weight(total_kg), delta=f"{stats.get('avg_total_per_record',0):.1f}kg/registro")
    with c4:
        sc = Utils.calculate_sustainability_score(pd.DataFrame({
            "recyclable_kg":[stats.get("total_recyclable_kg",0)],
            "organic_kg":[stats.get("total_organic_kg",0)],
            "waste_kg":[stats.get("total_waste_kg",0)],
        }))
        lab = "Excelente" if sc>=70 else "Bom" if sc>=50 else "Melhorar"
        color = "normal" if sc>=70 else ("off" if sc>=50 else "inverse")
        st.metric("🌱 Score Sustentabilidade", f"{sc:.1f}/100", delta=lab, delta_color=color)


def view_advanced_data_entry(db: AdvancedDatabaseManager):
    st.subheader("📝 Lançamento Inteligente de Dados")
    stats = db.get_statistics()

    with st.expander("🔍 Filtros Avançados", expanded=False):
        all_data = db.load_measurements()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            apartments = ["Todos"] + (sorted(all_data["apartment"].unique()) if not all_data.empty else [])
            selected_apartment = st.selectbox("🏠 Apartamento", apartments, key="entry_filter_apartment")
        with col2:
            blocks = ["Todos"] + (sorted(all_data["block"].unique()) if not all_data.empty else [])
            selected_block = st.selectbox("🏢 Bloco", blocks, key="entry_filter_block")
        with col3:
            min_week = int(all_data["week"].min()) if not all_data.empty else 1
            max_week = int(all_data["week"].max()) if not all_data.empty else 52
            # Garantir limites válidos para o slider (min < max)
            if min_week >= max_week:
                max_week = min_week + 1
            week_range = st.slider("📅 Intervalo de Semanas", min_week, max_week, (min_week, max_week), key="entry_filter_week_range")
        with col4:
            if not all_data.empty:
                dmin = all_data["reference_date"].min().date()
                dmax = all_data["reference_date"].max().date()
                date_range = st.date_input("📆 Período", value=(dmin,dmax), min_value=dmin, max_value=dmax, format="DD/MM/YYYY", key="entry_filter_date_range")
            else:
                date_range = (date.today(), date.today())

    filters: Dict[str, Any] = {}
    if selected_apartment != "Todos": filters["apartment"] = selected_apartment
    if selected_block != "Todos": filters["block"] = selected_block
    if "week_range" in locals(): filters["week_range"] = week_range
    if isinstance(date_range, tuple) and len(date_range) == 2: filters["date_range"] = [str(date_range[0]), str(date_range[1])]

    df = db.load_measurements(filters)
    if df.empty:
        st.warning("📭 Nenhum dado encontrado com os filtros aplicados.")
    else:
        with st.expander("📊 Visão Rápida", expanded=False):
            render_quick_stats(stats)

    with st.form("advanced_entry_form", clear_on_submit=False):
        st.markdown("### 📋 Informações Básicas")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            week = st.number_input("📅 Semana", min_value=1, max_value=52, value=datetime.now().isocalendar()[1], help="Semana do ano (ISO)", key="entry_week")
        with c2:
            all_blocks_df = db.load_measurements()
            existing_blocks = sorted(all_blocks_df["block"].unique()) if not all_blocks_df.empty else []
            options = (existing_blocks or []) + [f"BLOCO {i:02d}" for i in range(1,31) if f"BLOCO {i:02d}" not in (existing_blocks or [])]
            block = st.selectbox("🏢 Bloco", options if options else [f"BLOCO {i:02d}" for i in range(1,31)], key="entry_block_select_main")
        with c3:
            apartment = st.text_input("🚪 Apartamento", placeholder="Ex: 101, 201A, Apto 302", key="entry_apartment_input")
            if apartment and not Utils.validate_apartment_format(apartment):
                st.error("⚠️ Formato inválido. Ex: 101, 201A, Apto 302")
        with c4:
            reference_date = st.date_input("📆 Data de Referência", value=date.today(), max_value=date.today(), format="DD/MM/YYYY", key="entry_reference_date")

        st.markdown("---")
        st.markdown("### ♻️ Dados de Resíduos")
        c5,c6,c7,c8 = st.columns(4)
        with c5: recyclable = st.number_input("♻️ Reciclável (kg)", min_value=0.0, max_value=AppConfig.MAX_WEIGHT_KG, step=0.1, format="%.2f", key="entry_recyclable")
        with c6: organic = st.number_input("🌱 Orgânico (kg)", min_value=0.0, max_value=AppConfig.MAX_WEIGHT_KG, step=0.1, format="%.2f", key="entry_organic")
        with c7: waste = st.number_input("🗑️ Rejeito (kg)", min_value=0.0, max_value=AppConfig.MAX_WEIGHT_KG, step=0.1, format="%.2f", key="entry_waste")
        with c8: participating = st.number_input("👥 Aderência (aptos)", min_value=0, max_value=AppConfig.MAX_APARTMENTS, step=1, key="entry_participating")

        total = recyclable + organic + waste
        if total > 0:
            st.markdown("### 📊 Cálculos Automáticos")
            e = EnvironmentalConstants()
            cA,cB,cC,cD = st.columns(4)
            with cA:
                co2 = recyclable*e.CO2_EMISSIONS["reciclavel"] + organic*e.CO2_EMISSIONS["organico"] + waste*e.CO2_EMISSIONS["rejeito"]
                st.info(f"🌍 **CO₂:** {co2:.2f} kg")
            with cB:
                cost = recyclable*e.DISPOSAL_COSTS["reciclavel"] + organic*e.DISPOSAL_COSTS["organico"] + waste*e.DISPOSAL_COSTS["rejeito"]
                st.info(f"💰 **Custo:** {Utils.format_currency(cost)}")
            with cC:
                revenue = recyclable*e.RECYCLING_REVENUE["reciclavel"] + organic*e.RECYCLING_REVENUE["organico"]
                st.info(f"💚 **Economia:** {Utils.format_currency(revenue)}")
            with cD:
                water = recyclable*e.WATER_SAVINGS["reciclavel"] + organic*e.WATER_SAVINGS["organico"]
                st.info(f"💧 **Água Poupada:** {water:.1f} L")

        notes = st.text_area("📝 Observações (opcional)", placeholder="Comentários sobre a coleta, condições especiais, etc.", max_chars=500, key="entry_notes")

        submitted = st.form_submit_button("💾 Salvar Dados", type="primary", use_container_width=True)
        if submitted:
            errors = []
            if not apartment.strip(): errors.append("Apartamento é obrigatório.")
            elif not Utils.validate_apartment_format(apartment): errors.append("Formato de apartamento inválido.")
            if total == 0: errors.append("Informe pelo menos um tipo de resíduo.")
            if total > AppConfig.MAX_WEIGHT_KG: errors.append(f"Peso total excede {AppConfig.MAX_WEIGHT_KG} kg.")

            if errors:
                for err in errors: st.error(f"❌ {err}")
            else:
                data = {
                    "week": int(week), "block": block, "apartment": apartment.strip(),
                    "recyclable_kg": float(recyclable), "organic_kg": float(organic), "waste_kg": float(waste),
                    "participating_apts": int(participating), "reference_date": str(reference_date), "notes": notes.strip()
                }
                ok = db.save_measurement(data)
                if ok:
                    st.success("✅ Dados salvos com sucesso!"); st.balloons()
                else:
                    st.error("❌ Erro ao salvar dados. Verifique e tente novamente.")



def view_advanced_dashboard(db: AdvancedDatabaseManager):
    st.subheader("📊 Dashboard Executivo (Pro)")

    # ----------------------- Filtros Avançados -----------------------
    with st.expander("🔍 Filtros Avançados", expanded=False):
        all_data = db.load_measurements()
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            apartments = ["Todos"] + (sorted(all_data["apartment"].unique()) if not all_data.empty else [])
            sel_apartment = st.selectbox("🏠 Apartamento", apartments, key="dashpro_filter_apartment")
        with c2:
            blocks = ["Todos"] + (sorted(all_data["block"].unique()) if not all_data.empty else [])
            sel_block = st.selectbox("🏢 Bloco", blocks, key="dashpro_filter_block")
        with c3:
            min_week = int(all_data["week"].min()) if not all_data.empty else 1
            max_week = int(all_data["week"].max()) if not all_data.empty else 52
            if min_week >= max_week:
                max_week = min_week + 1
            week_range = st.slider("📅 Semanas", min_week, max_week, (min_week, max_week), key="dashpro_week_range")
        with c4:
            if not all_data.empty:
                dmin = all_data["reference_date"].min().date()
                dmax = all_data["reference_date"].max().date()
                date_range = st.date_input("📆 Período", value=(dmin,dmax), min_value=dmin, max_value=dmax,
                                           format="DD/MM/YYYY", key="dashpro_date_range")
            else:
                date_range = (date.today(), date.today())

    filters: Dict[str, Any] = {}
    if sel_apartment != "Todos": filters["apartment"] = sel_apartment
    if sel_block != "Todos": filters["block"] = sel_block
    filters["week_range"] = week_range
    if isinstance(date_range, tuple) and len(date_range)==2:
        filters["date_range"] = [str(date_range[0]), str(date_range[1])]

    df = db.load_measurements(filters)
    if df.empty:
        st.warning("📭 Nenhum dado encontrado com os filtros aplicados.")
        return

    # ----------------------- KPI Wall com Sparklines -----------------------
    stats = db.get_statistics()
    render_quick_stats(stats)

    st.markdown("### 🚀 Painel Visual Pro")
    k1, k2, k3, k4 = st.columns(4)
    # Preparos
    ts_total = df.groupby("reference_date", as_index=False)["total_kg"].sum().sort_values("reference_date")
    ts_rec = df.groupby("reference_date", as_index=False)["recyclable_kg"].sum().sort_values("reference_date")
    ts_co2 = df.groupby("reference_date", as_index=False)["co2_emissions_kg"].sum().sort_values("reference_date")
    ts_part = df.groupby("reference_date", as_index=False)["participating_apts"].sum().sort_values("reference_date")

    import plotly.graph_objects as go

    def sparkline(series_x, series_y, title):
        fig = go.Figure(go.Scatter(x=series_x, y=series_y, mode="lines", line=dict(width=2)))
        fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=140, title=title, template="plotly_white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    with k1: st.plotly_chart(sparkline(ts_total["reference_date"], ts_total["total_kg"], "Total (kg)"), use_container_width=True, key="spark_total")
    with k2: st.plotly_chart(sparkline(ts_rec["reference_date"], ts_rec["recyclable_kg"], "Reciclável (kg)"), use_container_width=True, key="spark_rec")
    with k3: st.plotly_chart(sparkline(ts_co2["reference_date"], ts_co2["co2_emissions_kg"], "CO₂ (kg)"), use_container_width=True, key="spark_co2")
    with k4: st.plotly_chart(sparkline(ts_part["reference_date"], ts_part["participating_apts"], "Aderência (aptos)"), use_container_width=True, key="spark_part")

    st.markdown("---")

    # ----------------------- Abas Pro -----------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Evolução Avançada",
        "🧩 Mix & Composição",
        "📐 Pareto & Controle",
        "🌡️ Mapa de Calor",
        "🫧 Dispersão"
    ])

    # ---------- Tab 1: Evolução Avançada (área empilhada + MA + CO2 e range) ----------
    with tab1:
        temporal = df.groupby("week", as_index=False).agg(
            recyclable_kg=("recyclable_kg","sum"),
            organic_kg=("organic_kg","sum"),
            waste_kg=("waste_kg","sum"),
            co2_emissions_kg=("co2_emissions_kg","sum")
        ).sort_values("week")
        temporal["total"] = temporal["recyclable_kg"] + temporal["organic_kg"] + temporal["waste_kg"]
        temporal["ma4"] = temporal["total"].rolling(4, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temporal["week"], y=temporal["recyclable_kg"], stackgroup="one", name="Reciclável"))
        fig.add_trace(go.Scatter(x=temporal["week"], y=temporal["organic_kg"], stackgroup="one", name="Orgânico"))
        fig.add_trace(go.Scatter(x=temporal["week"], y=temporal["waste_kg"], stackgroup="one", name="Rejeito"))
        fig.add_trace(go.Scatter(x=temporal["week"], y=temporal["ma4"], name="Média Móvel (4)", mode="lines", line=dict(width=3)))
        fig.update_layout(
            template="plotly_white",
            title="Evolução por Semana (Área Empilhada + Média Móvel)",
            xaxis_title="Semana", yaxis_title="Peso (kg)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=520
        )
        st.plotly_chart(fig, use_container_width=True, key="evo_stack_ma")

        # CO2 em eixo secundário
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=temporal["week"], y=temporal["total"], name="Total (kg)"))
        fig2.add_trace(go.Scatter(x=temporal["week"], y=temporal["co2_emissions_kg"], name="CO₂ (kg)", mode="lines+markers", yaxis="y2"))
        fig2.update_layout(
            template="plotly_white",
            title="Total vs CO₂",
            xaxis_title="Semana",
            yaxis=dict(title="Total (kg)"),
            yaxis2=dict(title="CO₂ (kg)", overlaying="y", side="right"),
            height=520
        )
        st.plotly_chart(fig2, use_container_width=True, key="total_vs_co2")

    # ---------- Tab 2: Mix & Composição (100% barras + Treemap) ----------
    with tab2:
        comp = df.groupby("week", as_index=False).agg(
            recyclable=("recyclable_kg","sum"),
            organic=("organic_kg","sum"),
            waste=("waste_kg","sum")
        )
        comp["sum"] = comp["recyclable"] + comp["organic"] + comp["waste"]
        for c in ["recyclable","organic","waste"]:
            comp[c] = np.where(comp["sum"]>0, comp[c]/comp["sum"], 0)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=comp["week"], y=comp["recyclable"], name="Reciclável"))
        fig3.add_trace(go.Bar(x=comp["week"], y=comp["organic"], name="Orgânico"))
        fig3.add_trace(go.Bar(x=comp["week"], y=comp["waste"], name="Rejeito"))
        fig3.update_layout(
            barmode="stack", template="plotly_white", height=520,
            title="Composição Semanal (100%)", xaxis_title="Semana", yaxis_title="%",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig3, use_container_width=True, key="mix_100")

        # Treemap por Bloco > Apartamento (Reciclável)
        tree_df = df.groupby(["block","apartment"], as_index=False)["recyclable_kg"].sum()
        fig4 = px.treemap(tree_df, path=["block","apartment"], values="recyclable_kg",
                          title="Distribuição Reciclável por Bloco e Apartamento")
        fig4.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig4, use_container_width=True, key="tree_rec")

    # ---------- Tab 3: Pareto & Controle ----------
    with tab3:
        # Pareto por apartamento (reciclável)
        pareto = df.groupby("apartment", as_index=False)["recyclable_kg"].sum().sort_values("recyclable_kg", ascending=False)
        pareto["cumul"] = pareto["recyclable_kg"].cumsum()
        total_rec = pareto["recyclable_kg"].sum() or 1.0
        pareto["cumul_pct"] = pareto["cumul"] / total_rec

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=pareto["apartment"], y=pareto["recyclable_kg"], name="Reciclável (kg)"))
        fig5.add_trace(go.Scatter(x=pareto["apartment"], y=pareto["cumul_pct"], name="Acumulado (%)", yaxis="y2"))
        fig5.update_layout(
            template="plotly_white", title="Pareto - Reciclável por Apartamento",
            yaxis=dict(title="kg"), yaxis2=dict(title="%", overlaying="y", side="right", tickformat=".0%"),
            height=520, xaxis_tickangle=-45
        )
        st.plotly_chart(fig5, use_container_width=True, key="pareto_rec")

        # Controle (X-bar) do total por semana
        ctrl = df.groupby("week", as_index=False)["total_kg"].sum().sort_values("week")
        mean = ctrl["total_kg"].mean()
        std = ctrl["total_kg"].std(ddof=0) or 0.0
        ucl = mean + 3*std
        lcl = max(mean - 3*std, 0)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=ctrl["week"], y=ctrl["total_kg"], mode="lines+markers", name="Total (kg)"))
        for yv, name, color in [(mean, "Média", "gray"), (ucl, "UCL (+3σ)", "red"), (lcl, "LCL (-3σ)", "red")]:
            fig6.add_hline(y=yv, line_dash="dash", annotation_text=name)
        fig6.update_layout(template="plotly_white", title="Gráfico de Controle (X-bar) - Total por Semana",
                           xaxis_title="Semana", yaxis_title="Total (kg)", height=520)
        st.plotly_chart(fig6, use_container_width=True, key="control_chart")

    # ---------- Tab 4: Heatmap ----------
    with tab4:
        heat = df.groupby(["block","week"], as_index=False)["total_kg"].sum()
        pivot = heat.pivot_table(index="block", columns="week", values="total_kg", fill_value=0)
        fig7 = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis",
                         title="Mapa de Calor - Peso Total (kg): Bloco x Semana")
        fig7.update_layout(template="plotly_white", height=540)
        st.plotly_chart(fig7, use_container_width=True, key="heatmap_block_week")

    # ---------- Tab 5: Dispersão (bubble) ----------
    with tab5:
        agg = df.groupby(["apartment","block"], as_index=False).agg(
            total=("total_kg","sum"),
            recyclable=("recyclable_kg","sum"),
            participation=("participating_apts","sum")
        )
        agg["recycling_rate"] = np.where(agg["total"]>0, agg["recyclable"]/agg["total"], 0.0)
        fig8 = px.scatter(
            agg, x="recycling_rate", y="total", size="participation", color="block",
            hover_data=["apartment"], size_max=40,
            title="Dispersão - Taxa de Reciclagem vs Total (bubble = participação)"
        )
        fig8.update_layout(template="plotly_white", height=540, xaxis_tickformat=".0%",
                           xaxis_title="Taxa de Reciclagem", yaxis_title="Total (kg)")
        st.plotly_chart(fig8, use_container_width=True, key="scatter_bubble")
def view_enterprise_reports(db: AdvancedDatabaseManager):
    st.subheader("📁 Central de Relatórios Executivos")
    df = db.load_measurements()
    if df.empty:
        st.warning("📭 Nenhum dado disponível para gerar relatórios.")
        return
    report = EnterpriseReportGenerator(df)

    st.markdown("### ⚙️ Configurações do Relatório")
    c1,c2,c3 = st.columns(3)
    with c1: report_type = st.selectbox("📊 Tipo de Relatório", ["Executivo Completo","Resumo Gerencial","Análise Ambiental","Relatório Operacional"], key="rep_type")
    with c2: format_opt = st.selectbox("📄 Formato", ["Excel Avançado","PDF Profissional","Ambos"], key="rep_format")
    with c3: period = st.selectbox("📅 Período", ["Todos os dados","Último mês","Últimas 4 semanas","Período personalizado"], key="rep_period")

    if period == "Período personalizado":
        d1, d2 = st.columns(2)
        with d1: start_date = st.date_input("Data Inicial", value=df["reference_date"].min().date(), key="rep_start")
        with d2: end_date = st.date_input("Data Final", value=df["reference_date"].max().date(), key="rep_end")
        df = df[(df["reference_date"] >= pd.Timestamp(start_date)) & (df["reference_date"] <= pd.Timestamp(end_date))]
        report = EnterpriseReportGenerator(df)

    st.markdown("### 👁️ Preview do Relatório")
    summary = report.calculate_comprehensive_summary()
    if not summary.empty:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("🏠 Apartamentos", summary.shape[0])
        with c2: st.metric("⚖️ Total Processado", Utils.format_weight(summary["total_kg_sum"].sum()))
        with c3: st.metric("💰 Receita Total", Utils.format_currency(summary["recycling_revenue_brl_sum"].sum()))
        with c4: st.metric("🌱 Score Médio", f"{summary['sustainability_score'].mean():.1f}/100")
        show = summary.head(10)[["apartment","sustainability_score","recycling_rate_percent","total_kg_sum","recycling_revenue_brl_sum","sustainability_rank"]].copy()
        show.columns = ["Apartamento","Score Sustentabilidade","Taxa Reciclagem (%)","Total (kg)","Receita (R$)","Ranking"]
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("### 💾 Gerar Relatórios")
    g1,g2,g3 = st.columns(3)
    with g1:
        if st.button("📊 Gerar Excel", use_container_width=True, type="primary", key="btn_excel"):
            try:
                st.download_button("⬇️ Download Excel", report.generate_advanced_excel(), file_name=f"relatorio_ecocondominio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_excel")
                st.success("✅ Excel gerado!")
            except Exception as e:
                st.error(f"❌ Erro ao gerar Excel: {e}")
    with g2:
        if st.button("📄 Gerar PDF", use_container_width=True, type="secondary", key="btn_pdf"):
            try:
                st.download_button("⬇️ Download PDF", report.generate_professional_pdf(), file_name=f"relatorio_ecocondominio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", use_container_width=True, key="dl_pdf")
                st.success("✅ PDF gerado!")
            except Exception as e:
                st.error(f"❌ Erro ao gerar PDF: {e}")
    with g3:
        if st.button("📦 Gerar Ambos", use_container_width=True, key="btn_both"):
            try:
                excel = report.generate_advanced_excel(); pdf = report.generate_professional_pdf()
                cE, cP = st.columns(2)
                with cE: st.download_button("📊 Excel", excel, file_name=f"relatorio_ecocondominio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_excel2")
                with cP: st.download_button("📄 PDF", pdf, file_name=f"relatorio_ecocondominio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", use_container_width=True, key="dl_pdf2")
                st.success("✅ Ambos gerados!")
            except Exception as e:
                st.error(f"❌ Erro ao gerar relatórios: {e}")


def main():
    st.set_page_config(page_title=AppConfig.PAGE_TITLE, page_icon=AppConfig.PAGE_ICON, layout=AppConfig.LAYOUT, initial_sidebar_state="collapsed")
    st.markdown(get_modern_css(), unsafe_allow_html=True)

    AppConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = AppConfig.DATA_DIR / AppConfig.DB_NAME
    db = AdvancedDatabaseManager(str(db_path))

    render_app_header()
    t1,t2,t3 = st.tabs(["📝 Entrada de Dados","📊 Dashboard Executivo","📁 Relatórios Enterprise"])
    with t1: view_advanced_data_entry(db)
    with t2: view_advanced_dashboard(db)
    with t3: view_enterprise_reports(db)

    st.markdown("---")
    with st.expander("ℹ️ Informações do Sistema", expanded=False):
        st.markdown(f"**Banco:** `{db_path.name}` - Registros: {db.get_statistics().get('total_records',0):,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Erro crítico do sistema: {e}")
        logger.critical(f"Critical system error: {e}")
        st.info("🔄 Recarregue a página ou contate o suporte técnico.")