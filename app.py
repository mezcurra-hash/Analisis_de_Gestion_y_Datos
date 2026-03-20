import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import numpy as np

# PDF opcional — requiere: pip install reportlab
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
st.set_page_config(
    page_title="CEMIC · Tablero de Gestión",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="auto"
)

# ============================================================
# PALETA Y ESTILOS GLOBALES
# ============================================================
ACCENT      = "#00BFA5"   # Teal principal
ACCENT2     = "#FF6B6B"   # Rojo suave para negativo
ACCENT3     = "#FFB74D"   # Ámbar para neutro/advertencia
BLUE_LIGHT  = "#4FC3F7"
BLUE_DARK   = "#0277BD"
CARD_BG     = "#1E2130"
BORDER      = "#2D3250"
TEXT_MUTED  = "#8B93A7"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#CDD6F4"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Fondo general ── */
  .stApp {{ background-color: #13151F; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background-color: #171925 !important;
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stRadio label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {TEXT_MUTED} !important;
  }}

  /* ── Tarjetas KPI ── */
  .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 12px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
  }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {ACCENT}, {BLUE_LIGHT});
  }}
  .kpi-label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {TEXT_MUTED};
      margin-bottom: 6px;
  }}
  .kpi-value {{
      font-size: 32px;
      font-weight: 700;
      color: #CDD6F4;
      font-family: 'DM Mono', monospace;
      line-height: 1;
  }}
  .kpi-delta-pos {{
      font-size: 13px;
      font-weight: 500;
      color: {ACCENT};
      margin-top: 6px;
  }}
  .kpi-delta-neg {{
      font-size: 13px;
      font-weight: 500;
      color: {ACCENT2};
      margin-top: 6px;
  }}
  .kpi-delta-neu {{
      font-size: 13px;
      font-weight: 500;
      color: {ACCENT3};
      margin-top: 6px;
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      background: {CARD_BG};
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
      border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      color: {TEXT_MUTED};
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {ACCENT} !important;
      color: #13151F !important;
      font-weight: 700;
  }}

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

  /* ── Dividers ── */
  hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

  /* ── Títulos de sección ── */
  .section-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
  }}
  .section-title {{
      font-size: 22px;
      font-weight: 700;
      color: #CDD6F4;
  }}
  .section-subtitle {{
      font-size: 13px;
      color: {TEXT_MUTED};
      margin-bottom: 20px;
  }}

  /* ── Badge de período ── */
  .badge {{
      display: inline-block;
      background: rgba(0,191,165,0.15);
      color: {ACCENT};
      border: 1px solid rgba(0,191,165,0.3);
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 600;
  }}

  /* ── Expander ── */
  [data-testid="stExpander"] {{
      background: {CARD_BG};
      border: 1px solid {BORDER} !important;
      border-radius: 10px;
  }}

  /* ── Métricas nativas (fallback) ── */
  div[data-testid="stMetric"] {{
      background-color: {CARD_BG};
      border: 1px solid {BORDER};
      padding: 14px 18px;
      border-radius: 12px;
  }}

  /* ── KPI alerta (SLA bajo meta) ── */
  .kpi-card-alert {{
      background: {CARD_BG};
      border: 1px solid {ACCENT2} !important;
      border-radius: 12px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      animation: pulse-border 2s ease-in-out infinite;
  }}
  .kpi-card-alert::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {ACCENT2}, #FF3333);
  }}
  @keyframes pulse-border {{
      0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,107,107,0); }}
      50%       {{ box-shadow: 0 0 8px 2px rgba(255,107,107,0.25); }}
  }}
  .kpi-alert-msg {{
      font-size: 11px;
      color: {ACCENT2};
      font-weight: 700;
      margin-top: 5px;
  }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
         7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
MESES_FULL = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
              7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}

def fmt_fecha(fecha):
    return f"{MESES_FULL[fecha.month]} {fecha.year}"

def fmt_num(n, decimals=0):
    return f"{n:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def kpi_card(label, value, delta=None, delta_pct=None, prefix="", suffix="", alert=False):
    """Renderiza una tarjeta KPI custom con HTML. alert=True activa borde rojo."""
    val_str = f"{prefix}{fmt_num(value)}{suffix}"
    delta_html = ""
    if delta is not None:
        sign  = "+" if delta >= 0 else ""
        cls   = "kpi-delta-pos" if delta > 0 else ("kpi-delta-neg" if delta < 0 else "kpi-delta-neu")
        icon  = "▲" if delta > 0 else ("▼" if delta < 0 else "●")
        pct_s = f" ({sign}{delta_pct:.1f}%)" if delta_pct is not None else ""
        delta_html = f'<div class="{cls}">{icon} {sign}{fmt_num(delta)}{suffix}{pct_s}</div>'
    alert_html = '<div class="kpi-alert-msg">⚠️ Por debajo de la meta (90%)</div>' if alert else ""
    card_class = "kpi-card-alert" if alert else "kpi-card"
    return f"""
    <div class="{card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val_str}</div>
        {delta_html}
        {alert_html}
    </div>
    """

# ============================================================
# HELPERS — EXPORTACIÓN
# ============================================================
def generar_excel(df: pd.DataFrame, nombre_hoja: str = "Datos") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=nombre_hoja, index=True)
    return buf.getvalue()

def generar_pdf(df: pd.DataFrame, titulo: str = "Reporte CEMIC") -> bytes:
    if not PDF_OK:
        return b""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=20, rightMargin=20,
                            topMargin=30, bottomMargin=20)
    styles = getSampleStyleSheet()
    elements = []

    # Título
    elements.append(Paragraph(titulo, styles['Title']))
    elements.append(Spacer(1, 12))

    # Reset index para incluirlo como columna
    df_reset = df.reset_index()
    data = [list(df_reset.columns)] + df_reset.astype(str).values.tolist()

    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1E2130')),
        ('TEXTCOLOR',  (0,0), (-1,0), rl_colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 7),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [rl_colors.HexColor('#13151F'), rl_colors.HexColor('#1A1D2E')]),
        ('TEXTCOLOR',  (0,1), (-1,-1), rl_colors.HexColor('#CDD6F4')),
        ('GRID',       (0,0), (-1,-1), 0.3, rl_colors.HexColor('#2D3250')),
        ('ALIGN',      (1,1), (-1,-1), 'RIGHT'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t)
    doc.build(elements)
    return buf.getvalue()

def botones_exportacion(df: pd.DataFrame, nombre_archivo: str, titulo_pdf: str):
    """Renderiza los botones de descarga Excel y PDF lado a lado."""
    col_xl, col_pdf, _ = st.columns([1, 1, 4])
    excel_bytes = generar_excel(df, nombre_archivo[:31])
    col_xl.download_button(
        label="⬇️ Excel",
        data=excel_bytes,
        file_name=f"{nombre_archivo}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    if PDF_OK:
        pdf_bytes = generar_pdf(df, titulo_pdf)
        col_pdf.download_button(
            label="⬇️ PDF",
            data=pdf_bytes,
            file_name=f"{nombre_archivo}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        col_pdf.caption("PDF: instalar reportlab")

# ============================================================
# SESSION STATE — Filtro cruzado entre módulos
# ============================================================
if 'cross_servicio' not in st.session_state:
    st.session_state['cross_servicio'] = []
if 'cross_depto' not in st.session_state:
    st.session_state['cross_depto'] = []

def apply_plotly_defaults(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color="#CDD6F4"), x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False)
    return fig

# ============================================================
# SIDEBAR — NAVEGACIÓN
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <img src="https://cemic.edu.ar/assets/img/logo/logo-cemic.png" width="130"
             style="filter: brightness(1.1);">
        <div style="font-size:10px; letter-spacing:0.15em; color:{TEXT_MUTED};
                    text-transform:uppercase; margin-top:8px; font-weight:600;">
            Tablero de Gestión
        </div>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.selectbox(
        "MÓDULO",
        ["🏥  Oferta de Turnos", "🎧  Call Center", "📉  Ausentismo"],
        label_visibility="collapsed"
    )
    st.markdown("<hr>", unsafe_allow_html=True)

# Hint de reapertura cuando sidebar está colapsado
st.markdown(f"""
<div style="position:fixed; bottom:16px; left:16px; z-index:9999;
            background:{ACCENT}; color:#13151F; border-radius:20px;
            padding:6px 14px; font-size:12px; font-weight:700;
            cursor:pointer; opacity:0.85;"
     title="Presioná [ para abrir el menú">
    ☰ Menú — presioná [
</div>
""", unsafe_allow_html=True)

# ============================================================
# APP 1: OFERTA DE TURNOS
# ============================================================
if app_mode == "🏥  Oferta de Turnos":

    @st.cache_data(ttl=300)
    def cargar_datos_oferta():
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQHFwl-Dxn-Rw9KN_evkCMk2Er8lQqgZMzAtN4LuEkWcCeBVUNwgb8xeIFKvpyxMgeGTeJ3oEWKpMZj/pub?gid=1524527213&single=true&output=csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        for col in ['SEDE','DEPARTAMENTO','SERVICIO']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        df['PERIODO'] = pd.to_datetime(df['PERIODO'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['PERIODO'])

    try:
        df = cargar_datos_oferta()

        # ── Sidebar: controles ──────────────────────────────
        with st.sidebar:
            st.markdown(f"<div style='font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:8px;'>VISTA</div>", unsafe_allow_html=True)
            modo = st.radio("", ["📊  Global", "🆚  Comparativa"], label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            fechas = sorted(df['PERIODO'].unique())

            if "Global" in modo:
                meses_sel = st.multiselect("PERÍODO", fechas, default=[fechas[-1]], format_func=fmt_fecha)
                periodos_para_comparar = None
            else:
                st.markdown(f"<div style='font-size:11px;font-weight:700;color:{TEXT_MUTED};'>PERÍODOS A COMPARAR</div>", unsafe_allow_html=True)
                meses_sel = st.multiselect(
                    "Seleccioná hasta 4 períodos",
                    fechas,
                    default=fechas[-2:] if len(fechas) >= 2 else fechas,
                    format_func=fmt_fecha,
                    max_selections=4
                )
                periodos_para_comparar = meses_sel

            st.markdown("<hr>", unsafe_allow_html=True)

            with st.expander("🔍  Filtros"):
                filtro_tipo = st.radio("Modalidad", ["Todos","AP","ANP"], horizontal=True)
                depto = st.multiselect("Departamento", sorted(df['DEPARTAMENTO'].unique()),
                                       default=st.session_state['cross_depto'] if st.session_state['cross_depto'] else [])
                serv  = st.multiselect("Servicio",     sorted(df['SERVICIO'].unique()),
                                       default=st.session_state['cross_servicio'] if st.session_state['cross_servicio'] else [])
                sede  = st.multiselect("Sede",         sorted(df['SEDE'].unique()))
                if 'PROFESIONAL/EQUIPO' in df.columns:
                    prof = st.multiselect("Profesional", sorted(df['PROFESIONAL/EQUIPO'].astype(str).unique()))
                else:
                    prof = []
                # Guardar selección para filtro cruzado
                st.session_state['cross_servicio'] = serv
                st.session_state['cross_depto']    = depto
                if serv or depto:
                    st.caption(f"🔗 Filtro activo · se aplicará en Ausentismo")

            st.markdown("<hr>", unsafe_allow_html=True)

            cols_txt = df.select_dtypes(include=['object']).columns.tolist()
            cols_num = df.select_dtypes(include=['float','int']).columns.tolist()
            default_fila = ['SERVICIO'] if 'SERVICIO' in cols_txt else [cols_txt[0]]
            filas_sel = st.multiselect("AGRUPAR POR", cols_txt, default=default_fila)
            val_sel   = st.multiselect("MÉTRICAS", cols_num,
                                       default=['TURNOS_MENSUAL'] if 'TURNOS_MENSUAL' in cols_num else [cols_num[0]])

        # ── Título ──────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="section-title">🏥 Oferta de Turnos de Consultorio</span></div>', unsafe_allow_html=True)

        if not meses_sel or not filas_sel or not val_sel:
            st.info("Seleccioná al menos un período, un agrupador y una métrica.")
            st.stop()

        # ── Filtrado ─────────────────────────────────────────
        df_f = df[df['PERIODO'].isin(meses_sel)].copy()
        if filtro_tipo == "AP"  and 'TIPO_ATENCION' in df_f.columns: df_f = df_f[df_f['TIPO_ATENCION']=='AP']
        if filtro_tipo == "ANP" and 'TIPO_ATENCION' in df_f.columns: df_f = df_f[df_f['TIPO_ATENCION']=='ANP']
        if depto: df_f = df_f[df_f['DEPARTAMENTO'].isin(depto)]
        if serv:  df_f = df_f[df_f['SERVICIO'].isin(serv)]
        if sede:  df_f = df_f[df_f['SEDE'].isin(sede)]
        if prof and 'PROFESIONAL/EQUIPO' in df_f.columns:
            df_f = df_f[df_f['PROFESIONAL/EQUIPO'].isin(prof)]

        if df_f.empty:
            st.warning("No hay datos para los filtros seleccionados.")
            st.stop()

        # ════════════════════════════════════════════════════
        # VISTA GLOBAL
        # ════════════════════════════════════════════════════
        if "Global" in modo:
            nombres = " · ".join([fmt_fecha(m) for m in sorted(meses_sel)])
            st.markdown(f'<div class="section-subtitle">Vista global · <span class="badge">{nombres}</span> · Modalidad: {filtro_tipo}</div>', unsafe_allow_html=True)

            # KPIs con delta vs período anterior al primero seleccionado
            primer_mes = min(meses_sel)
            idx_ant = fechas.index(primer_mes) - 1 if fechas.index(primer_mes) > 0 else None
            df_ant = df[df['PERIODO'] == fechas[idx_ant]] if idx_ant is not None else None

            cols_kpi = st.columns(len(val_sel))
            for i, metrica in enumerate(val_sel):
                val_actual = df_f[metrica].sum()
                delta = delta_pct = None
                if df_ant is not None and not df_ant.empty:
                    val_prev = df_ant[metrica].sum()
                    delta = val_actual - val_prev
                    delta_pct = (delta / val_prev * 100) if val_prev > 0 else 0
                label = metrica.replace("_"," ").title()
                cols_kpi[i].markdown(kpi_card(label, val_actual, delta, delta_pct), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            tab_graf, tab_tabla, tab_trend = st.tabs(["📊  Gráfico", "📄  Tabla dinámica", "📈  Tendencia"])

            with tab_graf:
                agrup_col = filas_sel[0]
                metrica_graf = val_sel[0]
                df_agg = df_f.groupby(agrup_col)[metrica_graf].sum().reset_index().sort_values(metrica_graf, ascending=False)

                fig = px.bar(
                    df_agg, x=agrup_col, y=metrica_graf,
                    text=metrica_graf,
                    color=metrica_graf,
                    color_continuous_scale=[[0, BLUE_DARK],[0.5, BLUE_LIGHT],[1, ACCENT]],
                )
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                  marker_line_width=0)
                fig.update_coloraxes(showscale=False)
                apply_plotly_defaults(fig, f"{metrica_graf.replace('_',' ').title()} por {agrup_col.title()}")
                fig.update_layout(height=460, xaxis_tickangle=-50)
                st.plotly_chart(fig, use_container_width=True)

            with tab_tabla:
                tabla = pd.pivot_table(df_f, index=filas_sel, values=val_sel,
                                       aggfunc='sum', margins=True, margins_name='TOTAL')
                tabla = tabla.fillna(0)
                st.dataframe(
                    tabla.style.format("{:,.0f}").bar(color=BLUE_LIGHT),
                    use_container_width=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                botones_exportacion(
                    tabla,
                    nombre_archivo=f"turnos_{'-'.join([fmt_fecha(m).replace(' ','_') for m in sorted(meses_sel)])}",
                    titulo_pdf=f"Oferta de Turnos — {nombres}"
                )

            with tab_trend:
                metrica_t = val_sel[0]

                # Serie completa — aplicar filtros pero NO filtro de período
                df_hist = df.copy()
                if filtro_tipo == "AP"  and 'TIPO_ATENCION' in df_hist.columns: df_hist = df_hist[df_hist['TIPO_ATENCION']=='AP']
                if filtro_tipo == "ANP" and 'TIPO_ATENCION' in df_hist.columns: df_hist = df_hist[df_hist['TIPO_ATENCION']=='ANP']
                if depto: df_hist = df_hist[df_hist['DEPARTAMENTO'].isin(depto)]
                if serv:  df_hist = df_hist[df_hist['SERVICIO'].isin(serv)]

                serie_completa = df_hist.groupby('PERIODO')[metrica_t].sum().reset_index().sort_values('PERIODO')

                # Separar meses cerrados de meses futuros
                # El mes en curso se incluye en el histórico (ya tiene datos cargados)
                # La proyección arranca desde el mes siguiente al actual
                hoy          = pd.Timestamp.today().normalize()
                corte        = (hoy.replace(day=1) + pd.DateOffset(months=1))  # Primer día del mes siguiente
                serie_pasada = serie_completa[serie_completa['PERIODO'] < corte]
                serie_futura = serie_completa[serie_completa['PERIODO'] >= corte]

                if len(serie_pasada) < 3:
                    st.info("Se necesitan al menos 3 meses cerrados para calcular la tendencia.")
                else:
                    # Regresión lineal SOLO sobre meses cerrados
                    x    = np.arange(len(serie_pasada))
                    y    = serie_pasada[metrica_t].values
                    coef = np.polyfit(x, y, 1)
                    poly = np.poly1d(coef)

                    # Proyección: 3 meses desde el último mes cerrado
                    N_PROJ      = 3
                    ultimo_cerrado = serie_pasada['PERIODO'].max()
                    fechas_proj = [ultimo_cerrado + pd.DateOffset(months=i+1) for i in range(N_PROJ)]
                    x_proj      = np.arange(len(serie_pasada), len(serie_pasada) + N_PROJ)
                    y_proj      = poly(x_proj)

                    fig_t = go.Figure()

                    # 1. Histórico real (meses cerrados)
                    fig_t.add_trace(go.Scatter(
                        x=serie_pasada['PERIODO'], y=serie_pasada[metrica_t],
                        name='Histórico (cerrado)', mode='lines+markers',
                        line=dict(color=BLUE_LIGHT, width=2),
                        marker=dict(size=6),
                        fill='tozeroy', fillcolor='rgba(79,195,247,0.1)',
                    ))

                    # 2. Oferta futura cargada (AP solamente — referencia visual, no entra en regresión)
                    if not serie_futura.empty:
                        fig_t.add_trace(go.Scatter(
                            x=serie_futura['PERIODO'], y=serie_futura[metrica_t],
                            name='Oferta futura (AP cargada)', mode='lines+markers',
                            line=dict(color=BLUE_LIGHT, width=2, dash='dot'),
                            marker=dict(size=6, symbol='circle-open'),
                            opacity=0.5,
                        ))

                    # 3. Línea de tendencia sobre el histórico cerrado
                    fig_t.add_trace(go.Scatter(
                        x=serie_pasada['PERIODO'], y=poly(x),
                        name='Tendencia', mode='lines',
                        line=dict(color=ACCENT3, width=2, dash='dot'),
                    ))

                    # 4. Proyección desde último mes cerrado hacia adelante
                    x_ext = [ultimo_cerrado] + fechas_proj
                    y_ext = [float(poly(len(serie_pasada)-1))] + list(y_proj)
                    fig_t.add_trace(go.Scatter(
                        x=x_ext, y=y_ext,
                        name='Proyección (regresión)', mode='lines+markers',
                        line=dict(color=ACCENT2, width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond'),
                        fill='tozeroy', fillcolor='rgba(255,107,107,0.08)',
                    ))

                    # 5. Línea vertical "Hoy"
                    fig_t.add_vline(
                        x=hoy.timestamp() * 1000,
                        line_width=1, line_dash="dot", line_color=ACCENT,
                        annotation_text=f"  Hoy ({MESES_FULL[hoy.month]} {hoy.year})",
                        annotation_font_color=ACCENT,
                        annotation_position="top right",
                    )

                    pendiente_dir = "creciente 📈" if coef[0] > 0 else "decreciente 📉"
                    apply_plotly_defaults(fig_t, f"Tendencia y proyección · {metrica_t.replace('_',' ').title()}")
                    fig_t.update_layout(height=420)
                    st.plotly_chart(fig_t, use_container_width=True)

                    # Tarjetas de proyección
                    cols_p = st.columns(N_PROJ)
                    for i, (fp, yp) in enumerate(zip(fechas_proj, y_proj)):
                        label_p = f"{MESES_FULL[fp.month]} {fp.year}"
                        cols_p[i].markdown(kpi_card(f"Proyección {label_p}", max(0, yp)), unsafe_allow_html=True)

                    n_meses_usados = len(serie_pasada)
                    st.markdown(f"""
                    <div style="margin-top:12px; padding:10px 14px; background:{CARD_BG};
                                border:1px solid {BORDER}; border-radius:8px; font-size:12px; color:{TEXT_MUTED};">
                        <b style="color:#CDD6F4;">Método:</b> Regresión lineal (mínimos cuadrados) ·
                        Basada en <b style="color:#CDD6F4;">{n_meses_usados} meses cerrados</b> ·
                        Tendencia <b style="color:#CDD6F4;">{pendiente_dir}</b> ·
                        Pendiente: <b style="color:#CDD6F4;">{coef[0]:+,.0f} turnos/mes</b>
                        <br><span style="font-size:11px;">⚠️ La oferta futura mostrada (línea punteada azul) incluye solo AP —
                        la ANP se agrega al inicio de cada mes.</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════
        # VISTA COMPARATIVA (multi-período)
        # ════════════════════════════════════════════════════
        else:
            if len(meses_sel) < 2:
                st.info("Seleccioná al menos 2 períodos para comparar.")
                st.stop()

            nombres = " vs ".join([fmt_fecha(m) for m in sorted(meses_sel)])
            st.markdown(f'<div class="section-subtitle">Comparativa · <span class="badge">{nombres}</span> · Modalidad: {filtro_tipo}</div>', unsafe_allow_html=True)

            metrica_kpi = val_sel[0]
            agrup_col   = filas_sel[0]

            # KPIs: último período vs penúltimo
            sorted_meses = sorted(meses_sel)
            df_ultimo   = df_f[df_f['PERIODO'] == sorted_meses[-1]]
            df_anterior = df_f[df_f['PERIODO'] == sorted_meses[-2]]

            cols_kpi = st.columns(len(val_sel))
            for i, metrica in enumerate(val_sel):
                va  = df_anterior[metrica].sum()
                vb  = df_ultimo[metrica].sum()
                dlt = vb - va
                pct = (dlt / va * 100) if va > 0 else 0
                label = metrica.replace("_"," ").title()
                cols_kpi[i].markdown(kpi_card(label, vb, dlt, pct), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab_comp, tab_var, tab_tabla = st.tabs(["📊  Comparación", "📈  Variación", "📄  Tabla"])

            with tab_comp:
                # Gráfico de barras agrupadas por período
                frames = []
                for m in sorted_meses:
                    tmp = df_f[df_f['PERIODO'] == m].groupby(agrup_col)[metrica_kpi].sum().reset_index()
                    tmp['Período'] = fmt_fecha(m)
                    frames.append(tmp)
                df_comp = pd.concat(frames)

                color_seq = [ACCENT, BLUE_LIGHT, ACCENT3, ACCENT2]
                fig = px.bar(df_comp, x=agrup_col, y=metrica_kpi, color='Período',
                             barmode='group', text=metrica_kpi,
                             color_discrete_sequence=color_seq)
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                  textfont_size=9, marker_line_width=0)
                apply_plotly_defaults(fig, f"Comparativa · {metrica_kpi.replace('_',' ').title()} por {agrup_col.title()}")
                fig.update_layout(height=480, xaxis_tickangle=-50)
                st.plotly_chart(fig, use_container_width=True)

            with tab_var:
                # Tabla de variación entre primer y último período
                g_base   = df_f[df_f['PERIODO']==sorted_meses[0]].groupby(agrup_col)[metrica_kpi].sum()
                g_actual = df_f[df_f['PERIODO']==sorted_meses[-1]].groupby(agrup_col)[metrica_kpi].sum()
                df_var   = pd.DataFrame({fmt_fecha(sorted_meses[0]): g_base, fmt_fecha(sorted_meses[-1]): g_actual}).fillna(0)
                df_var['Diferencia'] = df_var.iloc[:,1] - df_var.iloc[:,0]
                df_var['Var %']      = (df_var['Diferencia'] / df_var.iloc[:,0] * 100).replace([float('inf'),float('-inf')], 0)

                fig_var = px.bar(
                    df_var.reset_index().sort_values('Diferencia'),
                    x='Diferencia', y=agrup_col, orientation='h',
                    color='Diferencia',
                    color_continuous_scale=[[0, ACCENT2],[0.5, "#555"],[1, ACCENT]],
                    text='Diferencia'
                )
                fig_var.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                      marker_line_width=0)
                fig_var.update_coloraxes(showscale=False)
                apply_plotly_defaults(fig_var, f"Variación absoluta: {fmt_fecha(sorted_meses[0])} → {fmt_fecha(sorted_meses[-1])}")
                fig_var.update_layout(height=max(400, len(df_var)*22))
                st.plotly_chart(fig_var, use_container_width=True)

            with tab_tabla:
                st.dataframe(
                    df_var.style.format("{:,.0f}", subset=[fmt_fecha(sorted_meses[0]), fmt_fecha(sorted_meses[-1]), 'Diferencia'])
                               .format("{:.1f}%", subset=['Var %'])
                               .background_gradient(cmap='RdYlGn', subset=['Diferencia']),
                    use_container_width=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                botones_exportacion(
                    df_var,
                    nombre_archivo=f"comparativa_{fmt_fecha(sorted_meses[0]).replace(' ','_')}_vs_{fmt_fecha(sorted_meses[-1]).replace(' ','_')}",
                    titulo_pdf=f"Comparativa de Turnos — {nombres}"
                )

    except Exception as e:
        st.error(f"❌ Error en Oferta de Turnos: {e}")
        st.exception(e)


# ============================================================
# APP 2: CALL CENTER
# ============================================================
elif app_mode == "🎧  Call Center":

    @st.cache_data(ttl=300)
    def cargar_datos_cc():
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTOxpr7RRNTLGO96pUK8HJ0iy2ZHeqNpiR7OelleljCVoWPuJCO26q5z66VisWB76khl7Tmsqh5CqNC/pub?gid=0&single=true&output=csv"
        df = pd.read_csv(url, dtype=str).fillna("0")
        cols_num = ['RECIBIDAS_FIN','ATENDIDAS_FIN','PERDIDAS_FIN',
                    'RECIBIDAS_PREPAGO','ATENDIDAS_PREPAGO','PERDIDAS_PREPAGO',
                    'TURNOS_TOTAL_TEL','TURNOS_CONS_TEL','TURNOS_PRACT_TEL']
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].str.replace('.','',regex=False), errors='coerce').fillna(0)
        return df

    @st.cache_data(ttl=300)
    def cargar_datos_redes():
        # BD_REDES está en el mismo archivo que BD_CALLCENTER
        # Reemplazar REDES_GID con el gid real de la pestaña BD_REDES
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTOxpr7RRNTLGO96pUK8HJ0iy2ZHeqNpiR7OelleljCVoWPuJCO26q5z66VisWB76khl7Tmsqh5CqNC/pub?gid=734059738&single=true&output=csv"
        try:
            df = pd.read_csv(url, dtype=str).fillna('')
        except Exception as e:
            return pd.DataFrame(), str(e)
        df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=False)

        # Renombrar columnas del archivo al nombre estándar del código
        rename_map = {
            'MES'                      : 'MES',
            'INGRESADOS REDES'         : 'INGRESADOS_REDES',
            'ATENDIDOS OPERADOR'       : 'ATENDIDOS_REDES',
            'NO ATENDIDOS'             : 'NO_ATENDIDOS_REDES',
            'TURNOS PRÁCTICAS (AS)'    : 'TURNOS_PRACT_REDES',
            'TURNOS CONSULTORIOS (TS)' : 'TURNOS_CONS_REDES',
            'TOTAL TURNOS'             : 'TURNOS_TOTAL_REDES',
        }
        df = df.rename(columns=rename_map)

        df['FECHA_REAL'] = pd.to_datetime(df['MES'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['FECHA_REAL']).sort_values('FECHA_REAL')

        cols_num = ['INGRESADOS_REDES','ATENDIDOS_REDES','NO_ATENDIDOS_REDES',
                    'TURNOS_PRACT_REDES','TURNOS_CONS_REDES','TURNOS_TOTAL_REDES']
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace('.','',regex=False)
                                     .str.replace(',','',regex=False),
                    errors='coerce').fillna(0)
        return df, None

    def parsear_fecha(txt):
        if pd.isna(txt): return None
        t = str(txt).lower().strip().replace(".", "")
        m_map = {'ene':1,'feb':2,'mar':3,'abr':4,'may':5,'jun':6,
                 'jul':7,'ago':8,'sep':9,'oct':10,'nov':11,'dic':12,
                 'jan':1,'apr':4,'aug':8,'dec':12}
        p = t.replace("-"," ").split()
        if len(p) < 2: return None
        mes = m_map.get(p[0][:3])
        yr  = p[1] if len(p[1])==4 else "20"+p[1]
        return pd.Timestamp(year=int(yr), month=mes, day=1) if mes else None

    # Corte de sistema: antes de Jul-2024 los turnos estaban unificados
    CORTE_SISTEMAS = pd.Timestamp('2024-07-01')

    try:
        df_tel   = cargar_datos_cc()
        df_tel['FECHA_REAL'] = df_tel['MES'].apply(parsear_fecha)
        df_tel = df_tel.dropna(subset=['FECHA_REAL']).sort_values('FECHA_REAL')
        df_tel['TOTAL_LLAMADAS']  = df_tel['RECIBIDAS_FIN']  + df_tel['RECIBIDAS_PREPAGO']
        df_tel['TOTAL_ATENDIDAS'] = df_tel['ATENDIDAS_FIN']  + df_tel['ATENDIDAS_PREPAGO']
        df_tel['TOTAL_PERDIDAS']  = df_tel['PERDIDAS_FIN']   + df_tel['PERDIDAS_PREPAGO']
        df_tel['SLA']             = (df_tel['TOTAL_ATENDIDAS'] / df_tel['TOTAL_LLAMADAS'] * 100).fillna(0)

        df_red, redes_error = cargar_datos_redes()
        if redes_error:
            st.warning(f"⚠️ No se pudo cargar BD_REDES (¿publicaste la hoja?): {redes_error}")
        if not df_red.empty and 'ATENDIDOS_REDES' in df_red.columns and 'INGRESADOS_REDES' in df_red.columns:
            df_red['SLA_REDES'] = (df_red['ATENDIDOS_REDES'] / df_red['INGRESADOS_REDES'] * 100).fillna(0)
        elif not df_red.empty:
            st.warning(f"⚠️ BD_REDES no tiene las columnas esperadas. Columnas encontradas: {list(df_red.columns)}")
            df_red = pd.DataFrame()  # Reset para evitar errores downstream

        with st.sidebar:
            st.markdown(f"<div style='font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:8px;'>VISTA</div>", unsafe_allow_html=True)
            modo = st.radio("", ["📅  Mensual", "📱  Redes Sociales", "🔄  Interanual"],
                            label_visibility="collapsed")
            st.markdown("<hr>", unsafe_allow_html=True)
            if "Mensual" in modo:
                segmento = st.selectbox("SEGMENTO TELÉFONO",
                                        ["Unificado","Solo Financiadores","Solo Prepago"])

        st.markdown('<div class="section-header"><span class="section-title">🎧 Call Center</span></div>',
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # VISTA MENSUAL — Teléfono + resumen combinado
        # ══════════════════════════════════════════════════════
        if "Mensual" in modo:
            fechas = sorted(df_tel['FECHA_REAL'].unique(), reverse=True)
            sel    = st.selectbox("Período", fechas,
                                  format_func=lambda x: f"{MESES_FULL[x.month]} {x.year}")

            d     = df_tel[df_tel['FECHA_REAL'] == sel].iloc[0]
            d_ant = df_tel[df_tel['FECHA_REAL'] < sel].sort_values('FECHA_REAL').iloc[-1] \
                    if len(df_tel[df_tel['FECHA_REAL'] < sel]) > 0 else None

            if segmento == "Solo Financiadores":
                rec, aten, perd = d['RECIBIDAS_FIN'], d['ATENDIDAS_FIN'], d['PERDIDAS_FIN']
                rec_a, aten_a, perd_a = (d_ant['RECIBIDAS_FIN'], d_ant['ATENDIDAS_FIN'],
                                          d_ant['PERDIDAS_FIN']) if d_ant is not None else (0,0,0)
            elif segmento == "Solo Prepago":
                rec, aten, perd = d['RECIBIDAS_PREPAGO'], d['ATENDIDAS_PREPAGO'], d['PERDIDAS_PREPAGO']
                rec_a, aten_a, perd_a = (d_ant['RECIBIDAS_PREPAGO'], d_ant['ATENDIDAS_PREPAGO'],
                                          d_ant['PERDIDAS_PREPAGO']) if d_ant is not None else (0,0,0)
            else:
                rec, aten, perd = d['TOTAL_LLAMADAS'], d['TOTAL_ATENDIDAS'], d['TOTAL_PERDIDAS']
                rec_a, aten_a, perd_a = (d_ant['TOTAL_LLAMADAS'], d_ant['TOTAL_ATENDIDAS'],
                                          d_ant['TOTAL_PERDIDAS']) if d_ant is not None else (0,0,0)

            sla = (aten / rec * 100) if rec > 0 else 0

            st.markdown(f'<div class="section-subtitle">📞 Teléfono · <span class="badge">{MESES_FULL[sel.month]} {sel.year}</span></div>',
                        unsafe_allow_html=True)

            def delta_or_none(v, va): return (v-va, (v-va)/va*100) if va > 0 else (None, None)

            c1,c2,c3,c4 = st.columns(4)
            d1, p1 = delta_or_none(rec, rec_a)
            d2, p2 = delta_or_none(aten, aten_a)
            d3, p3 = delta_or_none(perd, perd_a)
            c1.markdown(kpi_card("Llamadas Recibidas", rec, d1, p1), unsafe_allow_html=True)
            c2.markdown(kpi_card("Atendidas", aten, d2, p2), unsafe_allow_html=True)
            c3.markdown(kpi_card("Abandonadas", perd, d3, p3), unsafe_allow_html=True)
            c4.markdown(kpi_card("Nivel de Servicio", sla, suffix="%"), unsafe_allow_html=True)
            if sla < 90:
                c4.error(f"⚠️ Bajo meta — {sla:.1f}% < 90%")

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = go.Figure(go.Pie(
                    labels=['Atendidas','Abandonadas'], values=[aten, perd], hole=0.55,
                    marker=dict(colors=[ACCENT, ACCENT2],
                                line=dict(color='rgba(0,0,0,0)', width=0)),
                    textinfo='percent+label', insidetextorientation='radial'
                ))
                fig_pie.add_annotation(text=f"<b>{sla:.0f}%</b><br>SLA",
                                       x=0.5, y=0.5, showarrow=False,
                                       font=dict(size=18, color="#CDD6F4"))
                apply_plotly_defaults(fig_pie, "Nivel de atención — Teléfono")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                turnos_data = {'Concepto': ['Consultorios','Prácticas','Total'],
                               'Cantidad': [d['TURNOS_CONS_TEL'], d['TURNOS_PRACT_TEL'], d['TURNOS_TOTAL_TEL']]}
                fig_t = px.bar(pd.DataFrame(turnos_data), x='Concepto', y='Cantidad',
                               text='Cantidad', color='Concepto',
                               color_discrete_map={'Consultorios': BLUE_LIGHT,
                                                   'Prácticas': BLUE_DARK, 'Total': ACCENT})
                fig_t.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                    marker_line_width=0)
                fig_t.update_layout(showlegend=False)
                apply_plotly_defaults(fig_t, "Turnos gestionados por teléfono")
                st.plotly_chart(fig_t, use_container_width=True)

            # ── Resumen combinado Teléfono + Redes ──────────
            dr_sel = df_red[df_red['FECHA_REAL'] == sel]
            if not dr_sel.empty:
                dr = dr_sel.iloc[0]
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-subtitle">📊 Resumen combinado · Teléfono + Redes · <span class="badge">{MESES_FULL[sel.month]} {sel.year}</span></div>',
                            unsafe_allow_html=True)

                total_ing  = rec + dr['INGRESADOS_REDES']
                total_aten = aten + dr['ATENDIDOS_REDES']
                total_no   = perd + dr['NO_ATENDIDOS_REDES']
                sla_comb   = (total_aten / total_ing * 100) if total_ing > 0 else 0

                # Advertencia si cruzamos el corte de sistemas
                if sel < CORTE_SISTEMAS:
                    st.markdown(f"""
                    <div class="insight-box insight-box-amber">
                        ⚠️ <b>Nota metodológica:</b> Antes de Julio 2024 los canales estaban en un
                        sistema unificado. El comparativo combinado pre Jul-2024 puede no reflejar
                        la separación real entre teléfono y redes.
                    </div>
                    """, unsafe_allow_html=True)

                cb1, cb2, cb3, cb4 = st.columns(4)
                cb1.markdown(kpi_card("📞+📱 Total Contactos", total_ing), unsafe_allow_html=True)
                cb2.markdown(kpi_card("✅ Total Atendidos", total_aten), unsafe_allow_html=True)
                cb3.markdown(kpi_card("❌ Total No Atendidos", total_no), unsafe_allow_html=True)
                cb4.markdown(kpi_card("📊 SLA Combinado", sla_comb, suffix="%"), unsafe_allow_html=True)

                # Barras comparativas teléfono vs redes
                st.markdown("<br>", unsafe_allow_html=True)
                df_comp_bar = pd.DataFrame({
                    'Canal'    : ['Teléfono','Teléfono','Redes','Redes'],
                    'Tipo'     : ['Atendidos','No Atendidos','Atendidos','No Atendidos'],
                    'Cantidad' : [aten, perd, dr['ATENDIDOS_REDES'], dr['NO_ATENDIDOS_REDES']],
                })
                fig_comp = px.bar(df_comp_bar, x='Canal', y='Cantidad', color='Tipo',
                                  barmode='stack', text='Cantidad',
                                  color_discrete_map={'Atendidos': ACCENT, 'No Atendidos': ACCENT2})
                fig_comp.update_traces(texttemplate='%{text:,.0f}', textposition='inside',
                                       marker_line_width=0)
                apply_plotly_defaults(fig_comp, "Contactos por canal — comparativa")
                fig_comp.update_layout(height=320)
                st.plotly_chart(fig_comp, use_container_width=True)

                # Turnos combinados (solo desde Jul-2024)
                if sel >= CORTE_SISTEMAS:
                    st.markdown("<br>", unsafe_allow_html=True)
                    turnos_comb = pd.DataFrame({
                        'Concepto' : ['Cons. Tel','Cons. Redes','Práct. Tel','Práct. Redes','Total Tel','Total Redes'],
                        'Cantidad' : [d['TURNOS_CONS_TEL'], dr['TURNOS_CONS_REDES'],
                                      d['TURNOS_PRACT_TEL'], dr['TURNOS_PRACT_REDES'],
                                      d['TURNOS_TOTAL_TEL'], dr['TURNOS_TOTAL_REDES']],
                        'Canal'    : ['Teléfono','Redes','Teléfono','Redes','Teléfono','Redes'],
                    })
                    fig_tc = px.bar(turnos_comb, x='Concepto', y='Cantidad', color='Canal',
                                    text='Cantidad', barmode='group',
                                    color_discrete_map={'Teléfono': BLUE_LIGHT, 'Redes': ACCENT3})
                    fig_tc.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                         marker_line_width=0)
                    apply_plotly_defaults(fig_tc, "Turnos por canal")
                    fig_tc.update_layout(height=320)
                    st.plotly_chart(fig_tc, use_container_width=True)

            # ── Evolución histórica ───────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(x=df_tel['FECHA_REAL'], y=df_tel['TOTAL_LLAMADAS'],
                name='Recibidas Tel', fill='tozeroy',
                line=dict(color=BLUE_LIGHT, width=2), fillcolor="rgba(79,195,247,0.15)"))
            fig_evo.add_trace(go.Scatter(x=df_tel['FECHA_REAL'], y=df_tel['TOTAL_ATENDIDAS'],
                name='Atendidas Tel', fill='tozeroy',
                line=dict(color=ACCENT, width=2), fillcolor="rgba(0,191,165,0.2)"))
            if not df_red.empty:
                fig_evo.add_trace(go.Scatter(x=df_red['FECHA_REAL'], y=df_red['INGRESADOS_REDES'],
                    name='Ingresados Redes', line=dict(color=ACCENT3, width=2, dash='dot'),
                    mode='lines'))
                fig_evo.add_trace(go.Scatter(x=df_red['FECHA_REAL'], y=df_red['ATENDIDOS_REDES'],
                    name='Atendidos Redes', line=dict(color=ACCENT3, width=2),
                    mode='lines'))
            fig_evo.add_vline(x=CORTE_SISTEMAS.timestamp()*1000, line_width=1,
                              line_dash="dash", line_color=TEXT_MUTED,
                              annotation_text="  Jul-2024: separación de sistemas",
                              annotation_font_color=TEXT_MUTED,
                              annotation_position="top right")
            apply_plotly_defaults(fig_evo, "Evolución histórica de contactos — Teléfono y Redes")
            fig_evo.update_layout(height=300)
            st.plotly_chart(fig_evo, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # VISTA REDES SOCIALES
        # ══════════════════════════════════════════════════════
        elif "Redes" in modo:
            if df_red.empty:
                st.warning("No hay datos de redes cargados.")
                st.stop()

            fechas_r = sorted(df_red['FECHA_REAL'].unique(), reverse=True)
            sel_r    = st.selectbox("Período", fechas_r,
                                    format_func=lambda x: f"{MESES_FULL[x.month]} {x.year}")

            dr     = df_red[df_red['FECHA_REAL'] == sel_r].iloc[0]
            dr_ant = df_red[df_red['FECHA_REAL'] < sel_r].sort_values('FECHA_REAL').iloc[-1] \
                     if len(df_red[df_red['FECHA_REAL'] < sel_r]) > 0 else None

            ing  = dr['INGRESADOS_REDES']
            aten = dr['ATENDIDOS_REDES']
            no_a = dr['NO_ATENDIDOS_REDES']
            sla_r = (aten / ing * 100) if ing > 0 else 0

            ing_a  = dr_ant['INGRESADOS_REDES']  if dr_ant is not None else 0
            aten_a = dr_ant['ATENDIDOS_REDES']   if dr_ant is not None else 0
            no_a_a = dr_ant['NO_ATENDIDOS_REDES'] if dr_ant is not None else 0

            st.markdown(f'<div class="section-subtitle">📱 Redes Sociales · <span class="badge">{MESES_FULL[sel_r.month]} {sel_r.year}</span></div>',
                        unsafe_allow_html=True)

            if sel_r < CORTE_SISTEMAS:
                st.markdown(f"""
                <div class="insight-box insight-box-amber">
                    ⚠️ <b>Nota metodológica:</b> Antes de Julio 2024 los turnos de redes y teléfono
                    estaban en un sistema unificado. Los datos de turnos por canal no están disponibles
                    para este período.
                </div>
                """, unsafe_allow_html=True)

            def dn(v, va): return (v-va, (v-va)/va*100) if va > 0 else (None, None)

            c1,c2,c3,c4 = st.columns(4)
            d1,p1 = dn(ing, ing_a)
            d2,p2 = dn(aten, aten_a)
            d3,p3 = dn(no_a, no_a_a)
            c1.markdown(kpi_card("📱 Contactos Ingresados", ing, d1, p1), unsafe_allow_html=True)
            c2.markdown(kpi_card("✅ Atendidos por Operador", aten, d2, p2), unsafe_allow_html=True)
            c3.markdown(kpi_card("❌ No Atendidos", no_a, d3, p3), unsafe_allow_html=True)
            c4.markdown(kpi_card("📊 Nivel de Atención", sla_r, suffix="%"), unsafe_allow_html=True)
            if sla_r < 90:
                c4.error(f"⚠️ Bajo meta — {sla_r:.1f}% < 90%")

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = go.Figure(go.Pie(
                    labels=['Atendidos','No Atendidos'], values=[aten, no_a], hole=0.55,
                    marker=dict(colors=[ACCENT3, ACCENT2],
                                line=dict(color='rgba(0,0,0,0)', width=0)),
                    textinfo='percent+label', insidetextorientation='radial'
                ))
                fig_pie.add_annotation(text=f"<b>{sla_r:.0f}%</b><br>Atención",
                                       x=0.5, y=0.5, showarrow=False,
                                       font=dict(size=18, color="#CDD6F4"))
                apply_plotly_defaults(fig_pie, "Nivel de atención — Redes")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                if sel_r >= CORTE_SISTEMAS:
                    t_pract = dr['TURNOS_PRACT_REDES']
                    t_cons  = dr['TURNOS_CONS_REDES']
                    t_tot   = dr['TURNOS_TOTAL_REDES']
                    turnos_r = {'Concepto': ['Consultorios','Prácticas','Total'],
                                'Cantidad': [t_cons, t_pract, t_tot]}
                    fig_tr = px.bar(pd.DataFrame(turnos_r), x='Concepto', y='Cantidad',
                                    text='Cantidad', color='Concepto',
                                    color_discrete_map={'Consultorios': ACCENT3,
                                                        'Prácticas': '#FF8F00', 'Total': ACCENT})
                    fig_tr.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                         marker_line_width=0)
                    fig_tr.update_layout(showlegend=False)
                    apply_plotly_defaults(fig_tr, "Turnos gestionados por redes")
                    st.plotly_chart(fig_tr, use_container_width=True)
                else:
                    st.info("Datos de turnos por canal disponibles desde Julio 2024.")

            # Evolución histórica redes
            st.markdown("<hr>", unsafe_allow_html=True)
            fig_evo_r = go.Figure()
            fig_evo_r.add_trace(go.Scatter(x=df_red['FECHA_REAL'], y=df_red['INGRESADOS_REDES'],
                name='Ingresados', fill='tozeroy',
                line=dict(color=ACCENT3, width=2), fillcolor='rgba(255,183,77,0.15)'))
            fig_evo_r.add_trace(go.Scatter(x=df_red['FECHA_REAL'], y=df_red['ATENDIDOS_REDES'],
                name='Atendidos', fill='tozeroy',
                line=dict(color=ACCENT, width=2), fillcolor='rgba(0,191,165,0.2)'))
            fig_evo_r.add_trace(go.Scatter(x=df_red['FECHA_REAL'], y=df_red['NO_ATENDIDOS_REDES'],
                name='No Atendidos', line=dict(color=ACCENT2, width=2, dash='dot')))
            fig_evo_r.add_vline(x=CORTE_SISTEMAS.timestamp()*1000, line_width=1,
                                line_dash="dash", line_color=TEXT_MUTED,
                                annotation_text="  Jul-2024: separación de sistemas",
                                annotation_font_color=TEXT_MUTED, annotation_position="top right")
            apply_plotly_defaults(fig_evo_r, "Evolución histórica — Redes Sociales")
            fig_evo_r.update_layout(height=300)
            st.plotly_chart(fig_evo_r, use_container_width=True)

            # Evolución de turnos por redes (solo desde Jul-2024)
            df_red_post = df_red[df_red['FECHA_REAL'] >= CORTE_SISTEMAS]
            if not df_red_post.empty and 'TURNOS_TOTAL_REDES' in df_red_post.columns:
                st.markdown("<hr>", unsafe_allow_html=True)
                fig_trn = go.Figure()
                fig_trn.add_trace(go.Scatter(x=df_red_post['FECHA_REAL'],
                    y=df_red_post['TURNOS_CONS_REDES'], name='Consultorios',
                    line=dict(color=ACCENT3, width=2), mode='lines+markers'))
                fig_trn.add_trace(go.Scatter(x=df_red_post['FECHA_REAL'],
                    y=df_red_post['TURNOS_PRACT_REDES'], name='Prácticas',
                    line=dict(color='#FF8F00', width=2), mode='lines+markers'))
                fig_trn.add_trace(go.Scatter(x=df_red_post['FECHA_REAL'],
                    y=df_red_post['TURNOS_TOTAL_REDES'], name='Total',
                    line=dict(color=ACCENT, width=2, dash='dot'), mode='lines+markers'))
                apply_plotly_defaults(fig_trn, "Evolución de turnos gestionados por redes (desde Jul-2024)")
                fig_trn.update_layout(height=280)
                st.plotly_chart(fig_trn, use_container_width=True)

            with st.expander("Ver datos históricos de redes"):
                st.dataframe(df_red[['FECHA_REAL','INGRESADOS_REDES','ATENDIDOS_REDES',
                                     'NO_ATENDIDOS_REDES','TURNOS_PRACT_REDES',
                                     'TURNOS_CONS_REDES','TURNOS_TOTAL_REDES','SLA_REDES']]
                             .style.format({'INGRESADOS_REDES':'{:,.0f}','ATENDIDOS_REDES':'{:,.0f}',
                                            'NO_ATENDIDOS_REDES':'{:,.0f}','TURNOS_PRACT_REDES':'{:,.0f}',
                                            'TURNOS_CONS_REDES':'{:,.0f}','TURNOS_TOTAL_REDES':'{:,.0f}',
                                            'SLA_REDES':'{:.1f}%'}),
                             use_container_width=True)

        # ══════════════════════════════════════════════════════
        # VISTA INTERANUAL
        # ══════════════════════════════════════════════════════
        else:
            st.markdown(f'<div class="section-subtitle">Mismo mes en distintos años</div>',
                        unsafe_allow_html=True)

            canal_int = st.radio("Canal:", ["📞 Teléfono","📱 Redes","📊 Combinado"], horizontal=True)
            mes_nom   = st.selectbox("Mes a comparar", list(MESES_FULL.values()))
            m_num     = list(MESES_FULL.values()).index(mes_nom) + 1

            if "Teléfono" in canal_int:
                df_i = df_tel[df_tel['FECHA_REAL'].dt.month == m_num].copy()
                if df_i.empty:
                    st.warning("Sin datos históricos para este mes.")
                else:
                    df_i['AÑO'] = df_i['FECHA_REAL'].dt.year.astype(str)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_i['AÑO'], y=df_i['TOTAL_ATENDIDAS'],
                        name='Atendidas', marker_color=ACCENT,
                        text=df_i['TOTAL_ATENDIDAS'], texttemplate='%{text:,.0f}'))
                    fig.add_trace(go.Bar(x=df_i['AÑO'], y=df_i['TOTAL_PERDIDAS'],
                        name='Abandonadas', marker_color=ACCENT2,
                        text=df_i['TOTAL_PERDIDAS'], texttemplate='%{text:,.0f}'))
                    fig.add_trace(go.Scatter(x=df_i['AÑO'], y=df_i['SLA'],
                        name='SLA %', yaxis='y2',
                        line=dict(color=ACCENT3, width=3), mode='lines+markers+text',
                        text=df_i['SLA'].apply(lambda x: f"{x:.0f}%"),
                        textposition='top center'))
                    apply_plotly_defaults(fig, f"Teléfono — {mes_nom} · comparativa interanual")
                    fig.update_layout(barmode='group', height=400,
                        yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                    title='SLA %', color=ACCENT3))
                    st.plotly_chart(fig, use_container_width=True)

            elif "Redes" in canal_int:
                df_i = df_red[df_red['FECHA_REAL'].dt.month == m_num].copy()
                if df_i.empty:
                    st.warning("Sin datos de redes para este mes.")
                else:
                    df_i['AÑO'] = df_i['FECHA_REAL'].dt.year.astype(str)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_i['AÑO'], y=df_i['ATENDIDOS_REDES'],
                        name='Atendidos', marker_color=ACCENT3,
                        text=df_i['ATENDIDOS_REDES'], texttemplate='%{text:,.0f}'))
                    fig.add_trace(go.Bar(x=df_i['AÑO'], y=df_i['NO_ATENDIDOS_REDES'],
                        name='No Atendidos', marker_color=ACCENT2,
                        text=df_i['NO_ATENDIDOS_REDES'], texttemplate='%{text:,.0f}'))
                    fig.add_trace(go.Scatter(x=df_i['AÑO'], y=df_i['SLA_REDES'],
                        name='Atención %', yaxis='y2',
                        line=dict(color=ACCENT, width=3), mode='lines+markers+text',
                        text=df_i['SLA_REDES'].apply(lambda x: f"{x:.0f}%"),
                        textposition='top center'))
                    apply_plotly_defaults(fig, f"Redes — {mes_nom} · comparativa interanual")
                    fig.update_layout(barmode='group', height=400,
                        yaxis2=dict(overlaying='y', side='right', showgrid=False,
                                    title='Atención %', color=ACCENT))
                    st.plotly_chart(fig, use_container_width=True)

            else:  # Combinado
                df_it = df_tel[df_tel['FECHA_REAL'].dt.month == m_num].copy()
                df_ir = df_red[df_red['FECHA_REAL'].dt.month == m_num].copy()
                if df_it.empty and df_ir.empty:
                    st.warning("Sin datos para este mes.")
                else:
                    df_it['AÑO'] = df_it['FECHA_REAL'].dt.year.astype(str)
                    df_ir['AÑO'] = df_ir['FECHA_REAL'].dt.year.astype(str)
                    fig = go.Figure()
                    if not df_it.empty:
                        fig.add_trace(go.Bar(x=df_it['AÑO'], y=df_it['TOTAL_ATENDIDAS'],
                            name='Atendidas Tel', marker_color=ACCENT,
                            text=df_it['TOTAL_ATENDIDAS'], texttemplate='%{text:,.0f}'))
                    if not df_ir.empty:
                        fig.add_trace(go.Bar(x=df_ir['AÑO'], y=df_ir['ATENDIDOS_REDES'],
                            name='Atendidos Redes', marker_color=ACCENT3,
                            text=df_ir['ATENDIDOS_REDES'], texttemplate='%{text:,.0f}'))
                    if pd.Timestamp(f"{min(df_it['FECHA_REAL'].dt.year.min() if not df_it.empty else 9999, df_ir['FECHA_REAL'].dt.year.min() if not df_ir.empty else 9999)}-{m_num:02d}-01") < CORTE_SISTEMAS:
                        st.markdown(f"""
                        <div class="insight-box insight-box-amber">
                            ⚠️ Antes de Julio 2024 los canales estaban unificados — la comparativa
                            interanual puede no ser homogénea para años anteriores a 2024.
                        </div>
                        """, unsafe_allow_html=True)
                    apply_plotly_defaults(fig, f"Combinado — {mes_nom} · comparativa interanual")
                    fig.update_layout(barmode='group', height=400)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error en Call Center: {e}")
        st.exception(e)



# ============================================================
# APP 3: AUSENTISMO
# ============================================================
elif app_mode == "📉  Ausentismo":

    @st.cache_data(ttl=300)
    def cargar_ausencias():
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQHFwl-Dxn-Rw9KN_evkCMk2Er8lQqgZMzAtN4LuEkWcCeBVUNwgb8xeIFKvpyxMgeGTeJ3oEWKpMZj/pub?gid=2132722842&single=true&output=csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], dayfirst=True, errors='coerce')
        return df

    try:
        df_aus = cargar_ausencias()

        # Detectar columna de métrica principal
        if 'CONSULTORIOS_REALES' in df_aus.columns:
            col_target = 'CONSULTORIOS_REALES'
            label_target = "Consultorios Cancelados"
        elif 'DIAS_CAIDOS' in df_aus.columns:
            col_target = 'DIAS_CAIDOS'
            label_target = "Días Caídos"
        else:
            st.error("❌ No se encontró la columna de métrica (CONSULTORIOS_REALES o DIAS_CAIDOS).")
            st.stop()

        if df_aus[col_target].dtype == 'object':
            df_aus[col_target] = pd.to_numeric(
                df_aus[col_target].str.replace('.','',regex=False), errors='coerce').fillna(0)
        else:
            df_aus[col_target] = pd.to_numeric(df_aus[col_target], errors='coerce').fillna(0)

        with st.sidebar:
            st.markdown(f"<div style='font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:8px;'>FILTROS</div>", unsafe_allow_html=True)
            años = sorted(df_aus['FECHA_INICIO'].dt.year.dropna().unique())
            año_sel = st.selectbox("AÑO", años, index=len(años)-1)
            df_y = df_aus[df_aus['FECHA_INICIO'].dt.year == año_sel].copy()
            df_y['MES_NUM'] = df_y['FECHA_INICIO'].dt.month
            meses_disp = sorted(df_y['MES_NUM'].dropna().unique())
            meses_sel  = st.multiselect("MES(ES)", meses_disp,
                                        default=meses_disp,
                                        format_func=lambda x: MESES_FULL.get(x, x))
            if meses_sel: df_y = df_y[df_y['MES_NUM'].isin(meses_sel)]

            st.markdown("<hr>", unsafe_allow_html=True)
            # Filtro cruzado — leer session_state de Turnos
            cross_srv = st.session_state.get('cross_servicio', [])
            cross_dpt = st.session_state.get('cross_depto', [])
            if cross_srv or cross_dpt:
                st.markdown(f"""
                <div style="background:rgba(0,191,165,0.1);border:1px solid rgba(0,191,165,0.3);
                            border-radius:8px;padding:8px 10px;font-size:11px;color:{ACCENT};margin-bottom:8px;">
                    🔗 <b>Filtro cruzado activo desde Turnos</b>
                </div>
                """, unsafe_allow_html=True)

            for col in ['DEPARTAMENTO','SERVICIO','MOTIVO','PROFESIONAL']:
                if col in df_y.columns:
                    opciones = sorted(df_y[col].astype(str).unique())
                    # Pre-cargar desde session_state si corresponde
                    if col == 'SERVICIO' and cross_srv:
                        default_val = [s for s in cross_srv if s in opciones]
                    elif col == 'DEPARTAMENTO' and cross_dpt:
                        default_val = [d for d in cross_dpt if d in opciones]
                    else:
                        default_val = []
                    sel = st.multiselect(col, opciones, default=default_val)
                    if sel: df_y = df_y[df_y[col].isin(sel)]

        if df_y.empty:
            st.warning("Sin datos para los filtros seleccionados.")
            st.stop()

        st.markdown('<div class="section-header"><span class="section-title">📉 Gestión de Ausentismo y Licencias</span></div>', unsafe_allow_html=True)
        meses_badge = " · ".join([MESES_FULL.get(m, str(m)) for m in sorted(meses_sel)])
        st.markdown(f'<div class="section-subtitle">Año {año_sel} · <span class="badge">{meses_badge}</span></div>', unsafe_allow_html=True)

        # KPIs
        total_cancel  = df_y[col_target].sum()
        n_eventos     = len(df_y)
        n_profs       = df_y['PROFESIONAL'].nunique() if 'PROFESIONAL' in df_y.columns else 0
        top_motivo    = df_y['MOTIVO'].mode()[0] if 'MOTIVO' in df_y.columns and not df_y.empty else "-"

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi_card(label_target, total_cancel), unsafe_allow_html=True)
        c2.markdown(kpi_card("Eventos / Licencias", n_eventos), unsafe_allow_html=True)
        c3.markdown(kpi_card("Profesionales", n_profs), unsafe_allow_html=True)

        # Para motivo principal usamos métrica nativa (string, no numérico)
        c4.metric("Motivo Principal", str(top_motivo))

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if 'MOTIVO' in df_y.columns:
                # Agrupar motivos menores en "Otros" — solo Top 8 visibles
                df_mot_full = df_y.groupby('MOTIVO')[col_target].sum().reset_index().sort_values(col_target, ascending=False)
                TOP_N = 8
                if len(df_mot_full) > TOP_N:
                    top8     = df_mot_full.head(TOP_N)
                    otros    = pd.DataFrame([{'MOTIVO': 'OTROS', col_target: df_mot_full.iloc[TOP_N:][col_target].sum()}])
                    df_mot   = pd.concat([top8, otros], ignore_index=True)
                else:
                    df_mot = df_mot_full

                COLOR_SEQ = [ACCENT, BLUE_LIGHT, ACCENT3, ACCENT2, "#B39DDB", "#80DEEA", "#FFCC80", "#F48FB1", "#AAAAAA"]
                fig_pie = go.Figure(go.Pie(
                    labels=df_mot['MOTIVO'],
                    values=df_mot[col_target],
                    hole=0.52,
                    marker=dict(colors=COLOR_SEQ[:len(df_mot)],
                                line=dict(color='rgba(0,0,0,0)', width=0)),
                    textinfo='percent',
                    hovertemplate='<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>',
                    insidetextorientation='radial',
                ))
                apply_plotly_defaults(fig_pie, "Distribución por motivo")
                fig_pie.update_layout(
                    legend=dict(
                        orientation="v",
                        x=1.02, y=0.5,
                        xanchor="left", yanchor="middle",
                        font=dict(size=12),
                        itemwidth=30,
                    ),
                    margin=dict(l=20, r=160, t=40, b=20),
                    height=420,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if 'SERVICIO' in df_y.columns:
                d_serv = df_y.groupby('SERVICIO')[col_target].sum().reset_index()\
                             .sort_values(col_target).tail(10)
                fig_hs = px.bar(d_serv, x=col_target, y='SERVICIO', orientation='h',
                                text=col_target, color=col_target,
                                color_continuous_scale=[[0, BLUE_DARK],[1, ACCENT2]])
                fig_hs.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                     marker_line_width=0)
                fig_hs.update_coloraxes(showscale=False)
                apply_plotly_defaults(fig_hs, "Top 10 servicios afectados")
                fig_hs.update_layout(height=420)
                st.plotly_chart(fig_hs, use_container_width=True)

        # Top profesionales — tooltip con desglose de motivos para cada uno
        if 'PROFESIONAL' in df_y.columns:
            st.markdown("<hr>", unsafe_allow_html=True)
            d_prof = df_y.groupby('PROFESIONAL')[col_target].sum().reset_index()\
                         .sort_values(col_target).tail(15)

            # Construir texto de tooltip con desglose de motivos por profesional
            if 'MOTIVO' in df_y.columns:
                def motivos_tooltip(prof_nombre):
                    sub = df_y[df_y['PROFESIONAL'] == prof_nombre]\
                              .groupby('MOTIVO')[col_target].sum()\
                              .sort_values(ascending=False)
                    lines = [f"  {m}: {int(v)}" for m, v in sub.items()]
                    return "<br>".join(lines)

                d_prof['tooltip_motivos'] = d_prof['PROFESIONAL'].apply(motivos_tooltip)

                fig_p = go.Figure(go.Bar(
                    x=d_prof[col_target],
                    y=d_prof['PROFESIONAL'],
                    orientation='h',
                    text=d_prof[col_target],
                    texttemplate='%{text:,.0f}',
                    textposition='outside',
                    marker=dict(
                        color=d_prof[col_target],
                        colorscale=[[0, BLUE_DARK],[0.5, BLUE_LIGHT],[1, ACCENT]],
                        line_width=0,
                    ),
                    customdata=d_prof['tooltip_motivos'],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"<b>Total {col_target.replace('_',' ').title()}:</b> %{{x:,.0f}}<br>"
                        "<br><b>Desglose por motivo:</b><br>"
                        "%{customdata}"
                        "<extra></extra>"
                    ),
                ))
            else:
                fig_p = px.bar(d_prof, x=col_target, y='PROFESIONAL', orientation='h',
                               text=col_target, color=col_target,
                               color_continuous_scale=[[0, BLUE_DARK],[0.5, BLUE_LIGHT],[1, ACCENT]])
                fig_p.update_traces(texttemplate='%{text:,.0f}', textposition='outside',
                                    marker_line_width=0)
                fig_p.update_coloraxes(showscale=False)

            apply_plotly_defaults(fig_p, "Top 15 profesionales con mayor ausentismo")
            fig_p.update_layout(height=max(420, len(d_prof)*30))
            st.plotly_chart(fig_p, use_container_width=True)

        # Evolución mensual
        st.markdown("<hr>", unsafe_allow_html=True)
        df_y_evo = df_y.copy()
        df_y_evo['MES_LABEL'] = df_y_evo['FECHA_INICIO'].dt.month.map(MESES_FULL)
        df_evo = df_y_evo.groupby('MES_NUM')[col_target].sum().reset_index()
        df_evo['MES_LABEL'] = df_evo['MES_NUM'].map(MESES_FULL)
        df_evo = df_evo.sort_values('MES_NUM')

        fig_evo = go.Figure(go.Bar(
            x=df_evo['MES_LABEL'], y=df_evo[col_target],
            text=df_evo[col_target], texttemplate='%{text:,.0f}',
            marker=dict(color=df_evo[col_target],
                        colorscale=[[0, BLUE_LIGHT],[1, ACCENT2]],
                        line_width=0)
        ))
        apply_plotly_defaults(fig_evo, f"Evolución mensual de {label_target}")
        fig_evo.update_layout(height=280)
        st.plotly_chart(fig_evo, use_container_width=True)

        with st.expander("📄 Ver registros detallados"):
            st.dataframe(df_y, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            botones_exportacion(
                df_y,
                nombre_archivo=f"ausentismo_{año_sel}",
                titulo_pdf=f"Ausentismo y Licencias — {año_sel}"
            )

    except Exception as e:
        st.error(f"❌ Error en Ausentismo: {e}")
        st.exception(e)
