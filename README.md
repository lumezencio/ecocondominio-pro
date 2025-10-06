EcoCondomínio Pro
EcoCondomínio Pro é uma plataforma orientada a dados para medir, auditar e otimizar a gestão de resíduos em condomínios. Transforma registros semanais de coleta em insights operacionais, impacto ambiental e economia, com dashboards executivos e relatórios premium (PDF/Excel) prontos para apresentações.

Stack: Python · Streamlit · Plotly · Pandas · SQLite · FPDF (fpdf2) · XlsxWriter · Kaleido


Funcionalidades

Entrada inteligente de dados

Validação em tempo real (formato de apartamento, limites, pesos).
Filtros avançados por apartamento / bloco / semana / período.


Dashboard Executivo

Sparklines (Total, Reciclável, CO₂, Aderência).
Evolução semanal (área empilhada + média móvel).
Total × CO₂ (eixo secundário), composição 100% por semana.
Treemap Bloco→Apartamento, Pareto, Gráfico de Controle (X-bar), Heatmap, Gráfico de bolhas.


Relatórios Empresariais

PDF Premium: capa, cartões executivos, páginas de gráficos, tabela de ranking zebrada, insights, rodapé paginado (compatível com Unicode).
Excel Avançado: dashboard, resumo por apartamento, dados brutos, análise de série temporal, rankings (com formatação condicional).


Arquitetura Robusta

SQLite com índices, restrições e triggers de auditoria (INSERT/UPDATE).
Backups automáticos, cache inteligente (TTL) e CSS moderno.




Arquitetura
ecocondominio-pro/
├─ ecocondominio_pro.py           # Aplicação Streamlit
├─ data/
│  ├─ ecocondominio.db            # Banco de dados SQLite (criado automaticamente)
│  ├─ backups/                    # Backups automáticos (com rotação)
│  └─ exports/                    # Arquivos exportados (se aplicável)
├─ assets/
│  ├─ DejaVuSans.ttf              # (opcional) Fonte Unicode para PDF
│  └─ DejaVuSans-Bold.ttf         # (opcional) Fonte Unicode para PDF
└─ app.log                        # Log da aplicação
Tabelas SQLite: measurements, settings, audit_log (com índices e triggers).

Instalação

Requer Python 3.10+

bashpython -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
Se preferir sem requirements.txt:
bashpip install streamlit pandas numpy plotly fpdf2 xlsxwriter kaleido pillow

Executar
bashstreamlit run ecocondominio_pro.py

A aplicação abre no seu navegador (veja a URL no terminal).


Exportar (PDF / Excel)
Abra "Relatórios Empresariais" dentro da aplicação:

Gerar PDF: relatório executivo premium.
Gerar Excel: pasta de trabalho com múltiplas planilhas contendo resumo, dados brutos, análise temporal e rankings.

Fontes Unicode (opcional) para acentos/símbolos perfeitos no PDF: coloque DejaVuSans.ttf e DejaVuSans-Bold.ttf na pasta assets/. Sem elas, a aplicação utiliza modo ASCII para evitar falhas.

Solução de Problemas

Gráficos não aparecem no PDF → pip install kaleido
Caractere inválido no PDF ("•", "—") → adicione fontes Unicode (acima) ou use o fallback ASCII já implementado.
"Not enough horizontal space…" → já tratado no layout (reset do cursor antes de cell(0, …) e multi_cell(0, …)). Se adicionar títulos/notas personalizadas muito longas, considere encurtá-las.


Publicar no GitHub (rápido)
bashgit config user.name "Luciano Henrique Mezencio"
git config user.email "lucianomezencio@gmail.com"
git config init.defaultBranch main
git init
git add .
git commit -m "chore: initial project import"
git branch -M main
git remote add origin https://github.com/mezen/ecocondominio-pro.git
git push -u origin main

Licença
Este projeto está sob licença própria. Consulte o arquivo LICENSE para mais detalhes.
Autor
Luciano Henrique Mezencio

GitHub: @mezen
Email: lucianomezencio@gmail.com
