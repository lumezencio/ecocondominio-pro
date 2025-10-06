# EcoCondomínio Pro

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**Plataforma inteligente para gestão e otimização de resíduos em condomínios**

[Funcionalidades](#funcionalidades) • [Instalação](#instalação) • [Como Usar](#como-usar) • [Documentação](#documentação)

</div>

---

## 📋 Sobre o Projeto

**EcoCondomínio Pro** é uma plataforma orientada a dados desenvolvida para medir, auditar e otimizar a gestão de resíduos em condomínios. O sistema transforma registros semanais de coleta em insights operacionais, análises de impacto ambiental e oportunidades de economia, oferecendo dashboards executivos e relatórios premium em PDF e Excel.

### 🎯 Principais Benefícios

- Monitoramento em tempo real da coleta de resíduos
- Identificação de oportunidades de redução de custos
- Cálculo automático de impacto ambiental (CO₂)
- Relatórios executivos profissionais
- Análise de aderência ao programa de reciclagem

---

## 🛠️ Tecnologias

**Stack Principal:**
- Python
- Streamlit
- Plotly
- Pandas
- SQLite
- FPDF (fpdf2)
- XlsxWriter
- Kaleido

---

## ✨ Funcionalidades

### 📊 Dashboard Executivo

- **Indicadores em Tempo Real**
  - Sparklines para Total, Reciclável, CO₂ e Aderência
  - Evolução semanal com média móvel
  - Análise de composição 100% por semana
  
- **Visualizações Avançadas**
  - Treemap hierárquico (Bloco → Apartamento)
  - Gráfico de Pareto
  - Gráfico de Controle (X-bar)
  - Heatmap de distribuição
  - Scatter plot com bolhas

### 📝 Entrada de Dados Inteligente

- Validação em tempo real
- Verificação de formato de apartamento
- Controle de limites e pesos
- Filtros avançados por apartamento, bloco, semana e período

### 📄 Relatórios Premium

**PDF Executivo:**
- Capa profissional
- Cartões de indicadores
- Páginas com gráficos de alta qualidade
- Tabelas de ranking formatadas
- Insights e recomendações
- Rodapé paginado
- Suporte total a Unicode

**Excel Avançado:**
- Dashboard com indicadores
- Resumo por apartamento
- Dados brutos exportados
- Análise de série temporal
- Rankings com formatação condicional

### 🔒 Arquitetura Robusta

- Banco de dados SQLite com índices otimizados
- Constraints e triggers de auditoria
- Backups automáticos com rotação
- Sistema de cache inteligente (TTL)
- Interface moderna com CSS customizado

---

## 📁 Estrutura do Projeto
