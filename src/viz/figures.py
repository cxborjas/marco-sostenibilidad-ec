from __future__ import annotations
import math
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuración de estilo profesional
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        pass  # Usar estilo por defecto

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = '#f8f9fa'
mpl.rcParams['axes.edgecolor'] = '#dee2e6'
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.color'] = '#dee2e6'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.8

# Paleta de colores moderna y profesional
COLOR_PALETTE = [
    '#667eea',  # Púrpura vibrante
    '#f093fb',  # Rosa claro
    '#4facfe',  # Azul cielo
    '#43e97b',  # Verde menta
    '#fa709a',  # Rosa salmón
    '#fee140',  # Amarillo brillante
    '#30cfd0',  # Turquesa
    '#a8edea',  # Aguamarina
    '#ff6b6b',  # Rojo coral
    '#feca57',  # Amarillo dorado
]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLOR_PALETTE)

MAX_KM_MONTHS: float | None = None


def _km_xlim(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    if not finite:
        return None
    limit = max(finite) * 1.05
    if MAX_KM_MONTHS:
        limit = min(limit, MAX_KM_MONTHS)
    return max(limit, 1.0)


def _fmt_int(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_percent(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    if v > 1.05:
        return f"{v:,.1f}%"
    return f"{v * 100:,.1f}%"


def _fmt_months(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:,.1f} meses"


def _fmt_float(value, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:,.{digits}f}"


def _is_finite_number(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _render_table(ax, rows: list[tuple[str, str]], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color='#2d3748')
    table = ax.table(cellText=rows, colLabels=["Indicador", "Valor"], loc="center", cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Estilo de encabezado
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#667eea')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
        cell.set_edgecolor('#5568d3')
    
    # Estilo de filas
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#e2e8f0')
            if j == 0:
                cell.set_text_props(weight='600', fontsize=9)
            else:
                cell.set_text_props(fontsize=9, color='#4a5568')


def _plot_km_multi(ax, km_map: dict[str, pd.DataFrame], subtitle: str) -> None:
    if not km_map:
        ax.text(0.5, 0.5, "KM no disponible", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(subtitle, fontsize=10, fontweight='600')
        return

    t_max_vals: list[float] = []
    colors = COLOR_PALETTE[:len(km_map)] if len(km_map) <= len(COLOR_PALETTE) else COLOR_PALETTE * (len(km_map) // len(COLOR_PALETTE) + 1)
    
    for idx, (label, km) in enumerate(km_map.items()):
        if km is None or km.empty:
            continue
        color = colors[idx]
        ax.step(km["t"], km["s"], where="post", label=str(label), linewidth=2.2, color=color)
        ax.fill_between(km["t"], 0, km["s"], step="post", alpha=0.15, color=color)
        if not km["t"].empty:
            t_max_vals.append(float(km["t"].max()))
        last = km.dropna(subset=["t", "s"]).tail(1)
        if not last.empty:
            x = float(last["t"].iloc[0])
            y = float(last["s"].iloc[0])
            ax.plot(x, y, 'o', markersize=6, color=color)
            ax.text(x, y, f"{label}: {_fmt_percent(y)}", fontsize=8, ha="left", va="center", fontweight='600')

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Meses desde inicio", fontweight='600')
    ax.set_ylabel("Supervivencia (Kaplan–Meier)", fontweight='600')
    ax.set_title(subtitle, fontsize=10, fontweight='600')
    ax.legend(fontsize=8, loc="best", frameon=True, shadow=True)
    limit = _km_xlim(t_max_vals)
    if limit:
        ax.set_xlim(0, limit)

def _trim_text(value: object, max_len: int = 60) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _normalize_geo_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())

def save_line_demografia(
    demo: pd.DataFrame,
    outpath: str,
    title: str,
    window_start: int,
    window_end: int,
    ruc_valid_start: int | None = None,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')
    ax.plot(demo["year"], demo["births_n"], marker='o', label="Creación", linewidth=2.5, markersize=6, color=COLOR_PALETTE[0])
    ax.plot(demo["year"], demo["closures_terminal_n"], marker='s', label="Cierre", linewidth=2.5, markersize=6, color=COLOR_PALETTE[1])
    for x, y in zip(demo["year"], demo["births_n"], strict=False):
        if pd.notna(x) and pd.notna(y):
            ax.text(x, y, _fmt_int(y), fontsize=7, ha="center", va="bottom", fontweight='600')
    for x, y in zip(demo["year"], demo["closures_terminal_n"], strict=False):
        if pd.notna(x) and pd.notna(y):
            ax.text(x, y, _fmt_int(y), fontsize=7, ha="center", va="top", fontweight='600')
    if not demo.empty:
        years = demo["year"].astype(int)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45, ha="right")

        # Calcular picos para incluirlos en la caja de leyenda
        births_peak = demo.loc[demo["births_n"].idxmax()] if demo["births_n"].max() > 0 else None
        closures_peak = demo.loc[demo["closures_terminal_n"].idxmax()] if demo["closures_terminal_n"].max() > 0 else None

    ax.set_xlim(window_start, window_end)
    ax.set_xlabel("Año", fontweight='600')
    ax.set_ylabel("Conteo (RUC)", fontweight='600')
    ax.set_title(title, fontweight='bold', color='#2d3748')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Leyenda con picos integrados
    leg = ax.legend(loc='best', frameon=True, shadow=True)
    peak_lines = []
    if not demo.empty:
        if births_peak is not None:
            peak_lines.append(f"Pico creación: {int(births_peak['year'])} ({_fmt_int(births_peak['births_n'])})")
        if closures_peak is not None:
            peak_lines.append(f"Pico cierre: {int(closures_peak['year'])} ({_fmt_int(closures_peak['closures_terminal_n'])})")
    if peak_lines:
        bbox = leg.get_frame()
        ax.annotate(
            "\n".join(peak_lines),
            xy=(0, 0), xycoords=bbox,
            xytext=(4, -4), textcoords="offset points",
            fontsize=7.5, fontweight='600', color='#4a5568',
            va="top", ha="left",
        )
    note_parts = [
        "Cierres=terminales observados; censura 31/12/2024.",
        "Conteos absolutos (RUC).",
    ]
    if isinstance(ruc_valid_start, int):
        note_parts.append(f"RUC con inicio válido: {ruc_valid_start:,}.")
    fig.text(0.01, 0.01, " ".join(note_parts), fontsize=7, ha="left", color='#718096')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_bar_cantones(cant: pd.DataFrame, outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    if cant.empty:
        ax.text(0.5, 0.5, "Sin datos de cantones", ha="center", va="center")
    else:
        data = cant.sort_values(["ruc_n", "establishments_n"], ascending=False).copy()
        positions = range(len(data))
        bars = ax.bar(positions, data["ruc_n"], color=COLOR_PALETTE[0], edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels(data["canton"], rotation=45, ha="right", fontweight='500')
        ax.set_ylabel("RUC (sociedades)", fontweight='600')
        ymax = max(data["ruc_n"].max(), 1)
        ax.set_ylim(0, ymax * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        shares = data.get("ruc_share")
        for bar, value, share in zip(bars, data["ruc_n"], shares, strict=False):
            label = _fmt_int(value)
            if _is_finite_number(share):
                label = f"{label} ({_fmt_percent(share)})"
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            offset = ymax * 0.02
            if y + offset > ax.get_ylim()[1]:
                ax.text(x, y - offset, label, ha="center", va="top", fontsize=8, color="white", fontweight='600')
            else:
                ax.text(x, y + offset, label, ha="center", va="bottom", fontsize=8, fontweight='600')

        total_ruc = None
        total_est = None
        if "ruc_share" in data.columns:
            ruc_share_sum = pd.to_numeric(data["ruc_share"], errors="coerce").sum()
            if _is_finite_number(ruc_share_sum) and ruc_share_sum > 0:
                total_ruc = int(round(data["ruc_n"].sum() / ruc_share_sum))
        if "establishments_share" in data.columns:
            est_share_sum = pd.to_numeric(data["establishments_share"], errors="coerce").sum()
            if _is_finite_number(est_share_sum) and est_share_sum > 0:
                total_est = int(round(data["establishments_n"].sum() / est_share_sum))

        note_parts = ["Unidad: RUC por cantón principal."]
        if total_ruc:
            note_parts.append(f"Denominador RUC: {total_ruc:,}.")
        if total_est:
            note_parts.append(f"Establecimientos (contexto): {total_est:,}.")
        fig.text(0.01, 0.01, " ".join(note_parts), fontsize=7, ha="left", color='#718096')

    ax.set_title(title, fontweight='bold', color='#2d3748')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_bar_macro(macro: pd.DataFrame, outpath: str, title: str):
    if not macro.empty:
        data = macro.copy()
        data["macro_sector"] = data["macro_sector"].astype("string").str.strip()
        data["ruc_n"] = pd.to_numeric(data.get("ruc_n"), errors="coerce").fillna(0).astype("int64")
        data["share"] = pd.to_numeric(data.get("share"), errors="coerce").fillna(0.0)
        data["macro_sector_label"] = data.get("macro_sector_label", "No informado").fillna("No informado")
        data = data.sort_values(["ruc_n", "macro_sector"], ascending=False).reset_index(drop=True)
    else:
        data = pd.DataFrame({"macro_sector": [], "ruc_n": [], "share": []})

    fig_height = max(5.5, 0.32 * len(data))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    fig.patch.set_facecolor('white')
    if data.empty or data["ruc_n"].sum() == 0:
        ax.text(0.5, 0.5, "Sin datos sectoriales", ha="center", va="center")
    else:
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(data))]
        ax.barh(data["macro_sector"], data["ruc_n"], color=colors, edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.set_ylabel("Macro-sector (letra CIIU)", fontweight='600')
        ax.set_xlabel("RUC (sociedades)", fontweight='600')
        ax.set_xlim(0, data["ruc_n"].max() * 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for y, (value, share) in enumerate(zip(data["ruc_n"], data["share"], strict=False)):
            label = _fmt_int(value)
            if _is_finite_number(share):
                label = f"{label} ({_fmt_percent(share)})"
            ax.text(value, y, f" {label}", va="center", fontsize=8, fontweight='600')

        note_parts = ["Unidad: RUC por macro-sector."]
        no_info_mask = data["macro_sector"].str.upper() == "NO INFORMADO"
        no_info_rows = data.loc[no_info_mask]
        if not no_info_rows.empty:
            ni_n = int(no_info_rows["ruc_n"].iloc[0])
            ni_s = no_info_rows["share"].iloc[0]
            ni_txt = f"No informado: {_fmt_int(ni_n)} ({_fmt_percent(ni_s)})"
            if _is_finite_number(ni_s) and float(ni_s) >= 0.1:
                ni_txt += " — interpretar shares sobre base informada."
            note_parts.append(ni_txt)
        else:
            note_parts.append("No informado: 0 (0.0%).")
        fig.text(0.01, 0.01, " ".join(note_parts), fontsize=7, ha="left", color='#718096')

    ax.set_title(title, fontweight='bold', color='#2d3748')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_bar_actividades(act: pd.DataFrame, outpath: str, title: str):
    if act.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de actividades", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    data = act.sort_values("ruc_n", ascending=True).copy()
    n = len(data)
    fig_height = max(5.5, 0.45 * n)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    fig.patch.set_facecolor('white')

    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n)]
    y_pos = range(n)
    bars = ax.barh(y_pos, data["ruc_n"], color=colors, edgecolor='white', linewidth=1.2, alpha=0.85)

    # Etiqueta: actividad dentro de la barra, valor+share fuera
    xmax = max(data["ruc_n"].max(), 1)
    for i, (bar, ciiu, actividad, value, share) in enumerate(zip(
        bars,
        data["ciiu"].astype("string"),
        data["actividad"].astype("string"),
        data["ruc_n"],
        data.get("share"),
        strict=False,
    )):
        # Nombre de actividad dentro de la barra (sin código, ya está en el eje Y)
        act_label = _trim_text(actividad, max_len=50)
        bar_w = bar.get_width()

        # Valor numérico justo después de la barra
        num_label = _fmt_int(value)
        if _is_finite_number(share):
            num_label = f"{num_label} ({_fmt_percent(share)})"

        if bar_w > xmax * 0.25:
            # Barra grande: actividad dentro, número fuera
            ax.text(bar_w * 0.02, i, f" {act_label}", va="center", ha="left",
                    fontsize=7, fontweight='600', color="white")
            ax.text(bar_w + xmax * 0.01, i, num_label, va="center", ha="left",
                    fontsize=7.5, fontweight='600', color="#2d3748")
        else:
            # Barra pequeña: número primero, luego actividad separada
            ax.text(bar_w + xmax * 0.01, i, f"{num_label}  ·  {act_label}",
                    va="center", ha="left", fontsize=7, fontweight='600', color="#4a5568")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["ciiu"].astype("string"), fontsize=8)
    ax.set_xlabel("RUC (sociedades)", fontweight='600')
    ax.set_ylabel("Código CIIU", fontweight='600')
    ax.set_xlim(0, xmax * 1.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(title, fontweight='bold', color='#2d3748')
    fig.text(0.01, 0.01, "Unidad: RUC por actividad (CIIU). Ranking sobre total RUC.", fontsize=7, ha="left", color='#718096')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_bar_cohortes(coh: pd.DataFrame, outpath: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if coh.empty or "cohort_5y" not in coh.columns:
        ax.text(0.5, 0.5, "Sin datos de cohortes", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    df = coh.copy()
    df["cohort_5y"] = df["cohort_5y"].astype("string")
    births_col = "births_n" if "births_n" in df.columns else None
    closures_col = "closures_terminal_n" if "closures_terminal_n" in df.columns else None
    if births_col is None and closures_col is None:
        ax.text(0.5, 0.5, "Sin nacimientos/cierres en cohortes", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    df = df.sort_values("cohort_5y").reset_index(drop=True)
    positions = range(len(df))
    width = 0.38
    if births_col:
        ax.bar([p - width / 2 for p in positions], df[births_col], width=width, label="Nacimientos")
    if closures_col:
        ax.bar([p + width / 2 for p in positions], df[closures_col], width=width, label="Cierres")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(df["cohort_5y"], rotation=45, ha="right")
    ax.set_ylabel("Conteo (RUC)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "Cohortes quinquenales por inicio de actividades.", fontsize=7, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_hist_duracion_cierres(ruc: pd.DataFrame, outpath: str, title: str, max_months: int | None = None):
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')
    df = ruc[(ruc["event"] == 1)].dropna(subset=["duration_months"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "Sin cierres observados", ha="center", va="center")
        ax.set_title(title, fontweight='bold', color='#2d3748')
        fig.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return

    x = df["duration_months"].astype("int64")
    if max_months is not None:
        x = x[x <= max_months]
    n = int(len(x))
    if n == 0:
        ax.text(0.5, 0.5, "Sin cierres en la ventana", ha="center", va="center")
        ax.set_title(title, fontweight='bold', color='#2d3748')
        fig.tight_layout()
        fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return

    # Binning a 12 meses (1 año)
    bin_width = 12
    max_x = max_months if max_months is not None else int(x.max())
    bins = list(range(0, max_x + bin_width, bin_width)) or [0, bin_width]
    counts_hist, bin_edges, patches = ax.hist(
        x, bins=bins, color=COLOR_PALETTE[0], alpha=0.8, edgecolor='white', linewidth=0.8,
    )

    # Estadísticos
    x_arr = x.values
    mean_v = float(np.mean(x_arr))
    median_v = float(np.median(x_arr))
    p25_v = float(np.percentile(x_arr, 25))
    p75_v = float(np.percentile(x_arr, 75))

    stat_lines = [
        (p25_v, "P25", "#43e97b", "dashdot"),
        (median_v, "Mediana", COLOR_PALETTE[8], "solid"),
        (mean_v, "Media", COLOR_PALETTE[2], "dashed"),
        (p75_v, "P75", "#fa709a", "dashdot"),
    ]
    ymax = float(max(counts_hist)) if len(counts_hist) else 1
    for val, label, color, ls in stat_lines:
        if val <= max_x:
            ax.axvline(val, linestyle=ls, color=color, linewidth=2, alpha=0.85)
            ax.text(
                val, ymax * 0.97, f" {label}: {int(round(val))}m ",
                fontsize=7.5, fontweight='700', color=color, va="top", ha="left",
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.85),
            )

    # Líneas de referencia temporal (1, 2, 5, 10 años)
    for ref, yr_label in [(12, "1a"), (24, "2a"), (60, "5a"), (120, "10a")]:
        if ref <= max_x:
            ax.axvline(ref, linestyle=":", color="#adb5bd", linewidth=1, alpha=0.7)
            ax.text(ref, ymax * 0.02, yr_label, fontsize=7, ha="center", va="bottom", color="#868e96")

    # Etiquetas en todas las barras
    for count_val, left_edge, right_edge in zip(counts_hist, bin_edges[:-1], bin_edges[1:]):
        if count_val > 0:
            pct = count_val / n * 100
            mid = (left_edge + right_edge) / 2
            ax.text(mid, count_val, f"{int(count_val)}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=6.5, fontweight='600', color='#495057')

    # CDF acumulada en eje secundario
    ax2 = ax.twinx()
    sorted_x = np.sort(x_arr)
    cdf_y = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    ax2.plot(sorted_x, cdf_y, color='#e67700', linewidth=2, alpha=0.8, label="% acumulado")
    ax2.set_ylabel("% acumulado de cierres", fontweight='600', color='#e67700')
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
    ax2.tick_params(axis='y', colors='#e67700')
    ax2.spines['right'].set_color('#e67700')
    ax2.spines['top'].set_visible(False)

    # Anotación sobre la CDF: % de cierres antes de 24m y 60m
    for ref_m in [24, 60]:
        if ref_m <= max_x and len(sorted_x):
            pct_ref = float((x_arr <= ref_m).sum()) / n
            ax2.annotate(
                f"{pct_ref:.0%} \u2264 {ref_m}m",
                xy=(ref_m, pct_ref), xycoords="data",
                xytext=(12, 6), textcoords="offset points",
                fontsize=7.5, fontweight='700', color='#e67700',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#e67700', alpha=0.85),
                arrowprops=dict(arrowstyle='->', color='#e67700', lw=1.2),
            )

    # Ejes y título
    ax.set_xlim(0, max_x)
    ax.set_xticks(range(0, max_x + 1, 25))
    ax.set_xlabel("Duración (meses) \u2014 sociedades cerradas", fontweight='600')
    ax.set_ylabel("Frecuencia", fontweight='600')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontweight='bold', color='#2d3748', pad=10)

    # Nota al pie (subtítulo + estadísticos)
    sub = f"Solo cierres observados (n={n:,}), bins={bin_width} meses, ventana 0\u2013{max_x}."
    note = (
        f"L\u00edneas verticales: P25={int(round(p25_v))}m, Mediana={int(round(median_v))}m, "
        f"Media={int(round(mean_v))}m, P75={int(round(p75_v))}m. "
        f"L\u00edneas punteadas grises: 1, 2, 5, 10 a\u00f1os. Curva naranja: CDF acumulada."
    )
    if n < 50:
        note += " Interpretar con prudencia por bajo n."
    fig.text(0.01, 0.005, f"{sub}  {note}", fontsize=6.5, ha="left", color='#718096')

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def save_km_plot(
    km: pd.DataFrame,
    outpath: str,
    title: str,
    counts: dict[str, int] | None = None,
    max_months: int | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    if km.empty:
        ax.text(0.5, 0.5, "KM no disponible (sin eventos)", ha="center", va="center")
    else:
        plot_km = km.copy()
        if max_months is not None:
            plot_km = plot_km[plot_km["t"] <= max_months]
        ax.step(plot_km["t"], plot_km["s"], where="post", linewidth=2.5, color=COLOR_PALETTE[0])
        ax.fill_between(plot_km["t"], 0, plot_km["s"], step="post", alpha=0.2, color=COLOR_PALETTE[0])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Meses desde inicio", fontweight='600')
        ax.set_ylabel("Supervivencia (Kaplan–Meier)", fontweight='600')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if max_months is not None:
            ax.set_xlim(0, max_months * 1.05)
        else:
            limit = _km_xlim([float(plot_km["t"].max())]) if not plot_km["t"].empty else None
            if limit:
                ax.set_xlim(0, limit)
        last = plot_km.dropna(subset=["t", "s"]).tail(1)
        if not last.empty:
            x = float(last["t"].iloc[0])
            y = float(last["s"].iloc[0])
            ax.plot(x, y, 'o', markersize=8, color=COLOR_PALETTE[0])
            ax.text(x, y, _fmt_percent(y), fontsize=9, ha="left", va="center", fontweight='600', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLOR_PALETTE[0], alpha=0.8))
    ax.set_title(title, fontweight='bold', color='#2d3748')
    if counts:
        n_total = counts.get("n_total")
        events_n = counts.get("events_n")
        censored_n = counts.get("censored_n")
        note = f"RUC incluidos: {_fmt_int(n_total)} · Eventos: {_fmt_int(events_n)} · Censurados: {_fmt_int(censored_n)}"
        if max_months is not None:
            note += f" · Eje X limitado a {max_months} meses (ventana de análisis)."
        fig.text(
            0.01,
            0.01,
            note,
            fontsize=8,
            ha="left",
            color='#718096'
        )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_km_multi(km_map: dict[str, pd.DataFrame], outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    if not km_map:
        ax.text(0.5, 0.5, "KM estratificado no disponible (umbral insuficiente)", ha="center", va="center")
    else:
        _plot_km_multi(ax, km_map, "")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_title(title, fontweight='bold', color='#2d3748')
    fig.text(0.01, 0.01, "Asociativo, no causal.", fontsize=8, ha="left", color='#718096')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_km_flags(km_flags: dict[str, dict[str, pd.DataFrame]], outpath: str, title: str) -> None:
    if not km_flags:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "KM por banderas no disponible", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    rows = len(km_flags)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3.5 * rows))
    if rows == 1:
        axes = [axes]

    for ax, (flag, km_map) in zip(axes, km_flags.items(), strict=False):
        _plot_km_multi(ax, km_map, f"KM por {flag}")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Asociativo, no causal.", fontsize=7, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_heatmap_placeholder(outpath: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "Heatmap cantonal no disponible", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_heatmap_cantones_geo(
    heatmap_table_path: str,
    geojson_path: str,
    outpath: str,
    title: str,
    province: str | None = None,
) -> None:
    try:
        import geopandas as gpd
    except Exception:
        save_heatmap_placeholder(outpath, title)
        return

    try:
        df = pd.read_csv(heatmap_table_path)
    except Exception:
        save_heatmap_placeholder(outpath, title)
        return

    if "canton" not in df.columns:
        save_heatmap_placeholder(outpath, title)
        return

    value_col = "ruc_share" if "ruc_share" in df.columns else "establishments_share"
    value_label = "Participacion RUC" if value_col == "ruc_share" else "Participacion establecimientos"

    df = df.copy()
    # Usar operaciones vectorizadas
    df["_join"] = df["canton"].str.strip().str.upper().str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception:
        save_heatmap_placeholder(outpath, title)
        return

    name_col = None
    for col in ["DPA_DESCAN", "DPA_DESCAN_", "DPA_DESCANM", "DPA_DES_CAN", "CANTON"]:
        if col in gdf.columns:
            name_col = col
            break

    if name_col is None:
        save_heatmap_placeholder(outpath, title)
        return

    if province and "DPA_DESPRO" in gdf.columns:
        prov_norm = _normalize_geo_name(province)
        # Usar operaciones vectorizadas
        gdf_prov = gdf["DPA_DESPRO"].str.strip().str.upper()
        gdf_prov = gdf_prov.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')
        gdf_prov = gdf_prov.str.split().str.join(' ')
        gdf = gdf[gdf_prov == prov_norm].copy()

    # Usar operaciones vectorizadas
    gdf_join = gdf[name_col].str.strip().str.upper()
    gdf_join = gdf_join.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')
    gdf["_join"] = gdf_join.str.split().str.join(' ')
    gdf = gdf.merge(df[["_join", value_col]], on="_join", how="left")

    if gdf[value_col].notna().sum() == 0:
        save_heatmap_placeholder(outpath, title)
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    gdf.plot(
        column=value_col,
        cmap="OrRd",
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": value_label, "shrink": 0.7},
        missing_kwds={"color": "#f0f0f0", "label": "Sin datos"},
        ax=ax,
    )
    label_rows = gdf[gdf[value_col].notna()].copy()
    if not label_rows.empty:
        label_rows["_pt"] = label_rows.geometry.representative_point()
        for _, row in label_rows.iterrows():
            name = str(row.get(name_col, "")).title().strip()
            value = row.get(value_col)
            if not name:
                continue
            label = f"{name}\n{_fmt_percent(value)}"
            ax.text(
                row["_pt"].x,
                row["_pt"].y,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
            )

    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_executive_kpi_card(kpis: dict[str, object], outpath: str, title: str):
    def _fmt_share_num(num, denom, share) -> str:
        if num is None or denom is None:
            return "N/A"
        return f"{_fmt_int(num)} / {_fmt_int(denom)} ({_fmt_percent(share)})"

    universe_rows = [
        ("Filas raw (establecimientos)", _fmt_int(kpis.get("raw_rows_establishments"))),
        ("RUC únicos", _fmt_int(kpis.get("unique_ruc_in_province"))),
        ("Estab/RUC mediana", _fmt_float(kpis.get("establishments_per_ruc_median"))),
        ("Estab/RUC p95", _fmt_float(kpis.get("establishments_per_ruc_p95"))),
        ("Faltantes críticos (promedio)", _fmt_percent(kpis.get("missing_critical_avg_share"))),
        ("Faltantes críticos (máx)", _fmt_percent(kpis.get("missing_critical_max_share"))),
        (
            "RUC multi-provincia",
            _fmt_share_num(
                kpis.get("multi_province_n"),
                kpis.get("multi_province_total"),
                kpis.get("multi_province_share"),
            ),
        ),
    ]

    demo_rows = [
        ("Nacimientos 2000-2024", _fmt_int(kpis.get("births_total_2000_2024"))),
        ("Cierres 2000-2024", _fmt_int(kpis.get("closures_terminal_total_2000_2024"))),
        ("Neto 2000-2024", _fmt_int(kpis.get("net_total_2000_2024"))),
        ("Nacimientos últimos 5 años", _fmt_int(kpis.get("births_last5"))),
        ("Cierres últimos 5 años", _fmt_int(kpis.get("closures_last5"))),
        ("Neto últimos 5 años", _fmt_int(kpis.get("net_last5"))),
    ]

    struct_rows = [
        ("Top3 cantonal (RUC)", _fmt_percent(kpis.get("top3_concentration_by_ruc_share"))),
        ("Top5 cantonal (RUC)", _fmt_percent(kpis.get("top5_concentration_by_ruc_share"))),
        ("Top3 cantonal (estab)", _fmt_percent(kpis.get("top3_concentration_by_establishments_share"))),
        ("Top5 cantonal (estab)", _fmt_percent(kpis.get("top5_concentration_by_establishments_share"))),
        (
            "Cantón líder",
            f"{kpis.get('leading_canton', 'N/A')} ({_fmt_percent(kpis.get('leading_canton_share'))})",
        ),
        (
            "Macro-sector líder",
            f"{kpis.get('leading_macro_sector', 'N/A')} ({_fmt_percent(kpis.get('leading_macro_sector_share'))})",
        ),
        ("No informado CIIU", _fmt_percent(kpis.get("macro_no_informado_share"))),
    ]

    surv_rows = [
        ("RUC incluidos", _fmt_int(kpis.get("unique_ruc_in_province"))),
        ("Eventos (cierres)", _fmt_int(kpis.get("events_n"))),
        ("Censurados", _fmt_int(kpis.get("censored_n"))),
        ("S(1) 12m", _fmt_percent(kpis.get("S_12m"))),
        ("S(2) 24m", _fmt_percent(kpis.get("S_24m"))),
        ("S(5) 60m", _fmt_percent(kpis.get("S_60m"))),
        ("S(10) 120m", _fmt_percent(kpis.get("S_120m"))),
        ("Mediana supervivencia", _fmt_months(kpis.get("median_survival_months"))),
        ("Cierre temprano (<24m)", _fmt_percent(kpis.get("early_closure_share_lt_24m"))),
        ("Periodo crítico", kpis.get("critical_period_bin") or "No alcanzado"),
    ]

    rows = [
        ("UNIVERSO Y CALIDAD", ""),
        *universe_rows,
        ("", ""),
        ("DEMOGRAFÍA", ""),
        *demo_rows,
        ("", ""),
        ("ESTRUCTURA", ""),
        *struct_rows,
        ("", ""),
        ("SUPERVIVENCIA (CORE)", ""),
        *surv_rows,
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Indicador", "Valor"], loc="center", cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Estilo de encabezado
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#667eea')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('#5568d3')
    
    # Estilo de secciones y filas
    section_indices = []
    for idx, row in enumerate(rows, 1):
        if row[1] == "" and row[0]:  # Es una cabecera de sección
            section_indices.append(idx)
    
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i in section_indices:
                # Cabeceras de sección
                cell.set_facecolor('#a8b2ff')
                cell.set_text_props(weight='bold', fontsize=10, color='#2d3748')
                cell.set_edgecolor('#667eea')
            elif rows[i-1][0] == "" and rows[i-1][1] == "":
                # Filas vacías
                cell.set_facecolor('white')
                cell.set_edgecolor('white')
            else:
                # Filas normales
                if i % 2 == 0:
                    cell.set_facecolor('#f8f9fa')
                else:
                    cell.set_facecolor('white')
                cell.set_edgecolor('#e2e8f0')
                if j == 0:
                    cell.set_text_props(weight='600', fontsize=9)
                else:
                    cell.set_text_props(fontsize=9, color='#4a5568')

    fig.suptitle(title, fontsize=15, fontweight='bold', color='#2d3748')
    window_start = kpis.get("window_start_year")
    window_end = kpis.get("window_end_year")
    censor_date = kpis.get("censor_date")
    fig.text(
        0.01,
        0.01,
        f"Periodo {window_start}-{window_end} · Censura {censor_date} · Asociativo, no causal.",
        fontsize=8,
        ha="left",
        color='#718096'
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_qc_dashboard(qc_raw: dict, qc_ruc: dict, outpath: str, title: str, qc_extra: dict | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Agregar título principal con estilo
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#2d3748', y=0.98)

    raw_rows = _fmt_int(qc_raw.get("raw_rows"))
    unique_ruc = _fmt_int(qc_raw.get("unique_ruc"))
    raw_rows_table = [
        ("Filas raw", raw_rows),
        ("RUC únicos", unique_ruc),
    ]

    domains = qc_raw.get("domains", {}) or {}
    domains_rows = [
        ("CLASE_CONTRIBUYENTE GEN/RMP/SIM", _fmt_percent(domains.get("CLASE_CONTRIBUYENTE_in_GEN_RMP_SIM_share"))),
        ("ESTADO_CONTRIBUYENTE ACTIVO/PASIVO/SUSP", _fmt_percent(domains.get("ESTADO_CONTRIBUYENTE_in_ACTIVO_PASIVO_SUSPENDIDO_share"))),
        ("ESTADO_ESTABLECIMIENTO ABI/CER", _fmt_percent(domains.get("ESTADO_ESTABLECIMIENTO_in_ABI_CER_share"))),
        ("TIPO_CONTRIBUYENTE PERSONA/SOCIEDAD", _fmt_percent(domains.get("TIPO_CONTRIBUYENTE_in_PERSONA_SOCIEDAD_share"))),
        ("CODIGO_JURISDICCION no vacío", _fmt_percent(domains.get("CODIGO_JURISDICCION_non_empty_share"))),
    ]

    ruc_rows = _fmt_int(qc_ruc.get("ruc_rows"))
    ruc_rows_table = [
        ("RUC en master", ruc_rows),
        ("Eventos", _fmt_int(qc_ruc.get("events_n"))),
        ("Censurados", _fmt_int(qc_ruc.get("censored_n"))),
        ("Duraciones negativas", _fmt_int(qc_ruc.get("negative_durations_n"))),
        ("Suspensión + reinicio", _fmt_int(qc_ruc.get("suspension_and_restart_n"))),
        ("Activo con cierre", _fmt_int((qc_ruc.get("state_vs_dates_audit", {}) or {}).get("activo_with_terminal_suspension_n"))),
        ("Suspendido sin fecha", _fmt_int((qc_ruc.get("state_vs_dates_audit", {}) or {}).get("suspendido_without_suspension_date_n"))),
    ]

    extra_rows = []
    if qc_extra:
        est_per_ruc = qc_extra.get("establishments_per_ruc", {}) or {}
        multi_prov = qc_extra.get("multi_province", {}) or {}
        extra_rows = [
            ("Estab/RUC media", _fmt_float(est_per_ruc.get("mean"))),
            ("Estab/RUC mediana", _fmt_float(est_per_ruc.get("median"))),
            ("Estab/RUC p95", _fmt_float(est_per_ruc.get("p95"))),
            ("RUC 1 estab", _fmt_percent(est_per_ruc.get("share_ruc_single_establishment"))),
            ("RUC multi-prov", _fmt_int(multi_prov.get("ruc_multi_province_in_province_n"))),
            ("% RUC multi-prov", _fmt_percent(multi_prov.get("ruc_multi_province_share"))),
        ]

    missing = qc_raw.get("missingness", {}) or {}
    top_missing = sorted(missing.items(), key=lambda item: item[1], reverse=True)[:8]
    missing_rows = [(col, _fmt_percent(val)) for col, val in top_missing]
    if not missing_rows:
        missing_rows = [("Sin datos", "N/A")]

    _render_table(axes[0, 0], raw_rows_table, "QC raw")
    _render_table(axes[0, 1], domains_rows, "Dominios esperados")
    if extra_rows:
        _render_table(axes[1, 0], ruc_rows_table + extra_rows, "QC RUC + contexto")
    else:
        _render_table(axes[1, 0], ruc_rows_table, "QC RUC")
    _render_table(axes[1, 1], missing_rows, "Top faltantes")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def save_metrics_dashboard(metrics: dict, outpath: str, title: str) -> None:
    run = metrics.get("run", {}) or {}
    dem = metrics.get("demography", {}) or {}
    geo = metrics.get("geography", {}) or {}
    sector = metrics.get("sector", {}) or {}
    surv = metrics.get("survival", {}) or {}

    run_rows = [
        ("Provincia", str(run.get("province", "N/A"))),
        ("Ventana", f"{run.get('window_start_year', 'N/A')} - {run.get('window_end_year', 'N/A')}"),
        ("Censor date", str(run.get("censor_date", "N/A"))),
        ("Raw filename", str((run.get("inputs", {}) or {}).get("raw_filename", "N/A"))),
        ("Nacimientos", _fmt_int(dem.get("births_total_2000_2024"))),
        ("Cierres terminales", _fmt_int(dem.get("closures_terminal_total_2000_2024"))),
        ("Netos", _fmt_int(dem.get("net_total_2000_2024"))),
    ]

    geo_rows = [
        ("Top3 RUC share", _fmt_percent(geo.get("top3_concentration_by_ruc_share"))),
        ("Top5 RUC share", _fmt_percent(geo.get("top5_concentration_by_ruc_share"))),
        ("Top3 estab share", _fmt_percent(geo.get("top3_concentration_by_establishments_share"))),
        ("Top5 estab share", _fmt_percent(geo.get("top5_concentration_by_establishments_share"))),
    ]

    leading = sector.get("leading_macro_sector", {}) or {}
    div = sector.get("diversification", {}) or {}
    sector_rows = [
        ("Macro líder", str(leading.get("letter", "N/A"))),
        ("Macro líder etiqueta", str(leading.get("label", "N/A"))),
        ("Macro líder share", _fmt_percent(leading.get("share"))),
        ("Top1 macro share", _fmt_percent(div.get("top1_macro_sector_share"))),
        ("HHI macro", _fmt_float(div.get("hhi_macro_sector"))),
        ("Macro efectivos", _fmt_float(div.get("effective_macro_sectors"))),
    ]

    surv_rows = [
        ("S 12m", _fmt_percent(surv.get("S_12m"))),
        ("S 24m", _fmt_percent(surv.get("S_24m"))),
        ("S 60m", _fmt_percent(surv.get("S_60m"))),
        ("S 120m", _fmt_percent(surv.get("S_120m"))),
        ("Mediana", _fmt_months(surv.get("median_survival_months"))),
        ("Cierres <24m", _fmt_percent(surv.get("early_closure_share_lt_24m"))),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Agregar título principal con estilo
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#2d3748', y=0.98)
    
    _render_table(axes[0, 0], run_rows, "Run + demografía")
    _render_table(axes[0, 1], geo_rows, "Geografía")
    _render_table(axes[1, 0], sector_rows, "Sector")
    _render_table(axes[1, 1], surv_rows, "Supervivencia")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
