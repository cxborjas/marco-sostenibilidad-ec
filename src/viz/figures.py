from __future__ import annotations
import math
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

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

def _fmt_num_share(num, share) -> str:
    if num is None:
        return "N/A"
    base = _fmt_int(num)
    if share is None:
        return base
    try:
        s = float(share)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(s):
        return base
    return f"{base} ({_fmt_percent(s)})"

def _fmt_rate(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v * 100:,.2f}%"


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

def _fmt_pvalue(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(v) or v < 0:
        return "N/A"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"

def _survival_at_time(km: pd.DataFrame, t: float) -> float:
    if km is None or km.empty or "t" not in km.columns or "s" not in km.columns:
        return float("nan")
    km2 = km[km["t"] <= t]
    if km2.empty:
        return 1.0
    return float(km2["s"].iloc[-1])

def _add_at_risk_table(ax, times: list[int], at_risk: dict[str, list[int]],
                       label_map: dict[str, str] | None = None,
                       row_order: list[str] | None = None,
                       y_offset: float | None = None) -> None:
    if not at_risk or not times:
        return
    order = row_order or list(at_risk.keys())
    order = [g for g in order if g in at_risk]
    if not order:
        return
    col_labels = [str(int(t)) for t in times]
    row_labels = [label_map.get(g, g) if label_map else g for g in order]
    cell_text = []
    for g in order:
        row_vals = at_risk.get(g, [])
        cell_text.append([_fmt_int(v) for v in row_vals])

    n_rows = len(order)
    if y_offset is None:
        y_offset = -0.35 - 0.08 * max(n_rows - 1, 0)
    height = 0.18 + 0.05 * max(n_rows - 1, 0)
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, y_offset, 1.0, height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)


def _is_finite_number(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _render_table(ax, rows: list[tuple[str, str]], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=16, color='#2d3748')
    n_rows = len(rows) + 1
    if n_rows >= 11:
        bbox = [0.0, 0.0, 1.0, 0.82]
        row_scale = 1.2
    else:
        bbox = [0.0, 0.0, 1.0, 0.9]
        row_scale = 1.5
    table = ax.table(
        cellText=rows,
        colLabels=["Indicador", "Valor"],
        loc="center",
        cellLoc='left',
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, row_scale)
    
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


def _pretty_label(raw: str) -> str:
    if raw is None:
        return "N/A"
    text = str(raw).replace("_", " ").strip()
    if not text:
        return "N/A"
    acronyms = {"ruc", "sri", "iess", "iva"}
    words = []
    for w in text.split():
        lw = w.lower()
        if lw in acronyms:
            words.append(lw.upper())
        elif w.isupper() and len(w) <= 4:
            words.append(w)
        else:
            words.append(w.capitalize())
    return " ".join(words)


def _plot_km_multi(ax, km_map: dict[str, pd.DataFrame], subtitle: str,
                   max_months: float | None = None,
                   label_prefix: str = "",
                   fill: bool = True,
                   fill_alpha: float = 0.15,
                   group_counts: dict[str, int] | None = None,
                   group_stats: dict[str, dict[str, int]] | None = None,
                   label_map: dict[str, str] | None = None,
                   show_last_point: bool = True,
                   milestone_times: list[int] | None = None,
                   milestone_label_time: int | None = None,
                   line_styles: list[str] | None = None,
                   legend_only: list[str] | None = None,
                   legend_only_suffix: str | None = None) -> None:
    if not km_map:
        ax.text(0.5, 0.5, "KM no disponible", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(subtitle, fontsize=10, fontweight='600')
        return

    t_max_vals: list[float] = []
    colors = COLOR_PALETTE[:len(km_map)] if len(km_map) <= len(COLOR_PALETTE) else COLOR_PALETTE * (len(km_map) // len(COLOR_PALETTE) + 1)
    
    def _build_label(label_raw: str) -> str:
        display_label = label_map.get(label_raw, label_raw) if label_map else label_raw
        if label_prefix:
            display_label = f"{label_prefix} {display_label}"
        if group_stats:
            stats = group_stats.get(label_raw)
            if stats:
                n_txt = _fmt_int(stats.get("n"))
                ev_txt = _fmt_int(stats.get("events"))
                cens_txt = _fmt_int(stats.get("censored"))
                display_label = f"{display_label} (n={n_txt}; ev={ev_txt}; cens={cens_txt})"
        elif group_counts:
            count = group_counts.get(label_raw)
            if count is not None:
                display_label = f"{display_label} (n={_fmt_int(count)})"
        return display_label

    for idx, (label, km) in enumerate(km_map.items()):
        if km is None or km.empty:
            continue
        color = colors[idx]
        label_raw = str(label)
        display_label = _build_label(label_raw)
        linestyle = line_styles[idx % len(line_styles)] if line_styles else "solid"
        ax.step(
            km["t"],
            km["s"],
            where="post",
            label=display_label,
            linewidth=2.4,
            color=color,
            linestyle=linestyle,
        )
        if fill:
            ax.fill_between(km["t"], 0, km["s"], step="post", alpha=fill_alpha, color=color)
        if not km["t"].empty:
            t_max_vals.append(float(km["t"].max()))
        if show_last_point:
            # Punto final: recortar a max_months para no desbordar
            if max_months:
                clip = km[km["t"] <= max_months].dropna(subset=["t", "s"]).tail(1)
            else:
                clip = km.dropna(subset=["t", "s"]).tail(1)
            if not clip.empty:
                x = float(clip["t"].iloc[0])
                y = float(clip["s"].iloc[0])
                ax.plot(x, y, 'o', markersize=6, color=color)
                ax.text(x, y, f" {_fmt_percent(y)}", fontsize=8, ha="left", va="center",
                        fontweight='600', color=color)

        if milestone_times:
            label_time = milestone_label_time or max(milestone_times)
            for t_ref in milestone_times:
                y_ref = _survival_at_time(km, t_ref)
                if not math.isfinite(y_ref):
                    continue
                ax.plot(t_ref, y_ref, 'o', markersize=4, color=color, alpha=0.9)
                if t_ref == label_time:
                    ax.text(t_ref, y_ref, f" {_fmt_percent(y_ref)}", fontsize=8, ha="left",
                            va="center", fontweight='600', color=color)

    if legend_only:
        suffix = f" ({legend_only_suffix})" if legend_only_suffix else ""
        for label_raw in legend_only:
            display_label = _build_label(str(label_raw)) + suffix
            ax.plot([], [], label=display_label, linestyle="dotted", color="#6c757d", linewidth=2.0)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Meses desde inicio", fontweight='600')
    ax.set_ylabel("Supervivencia (Kaplan–Meier)", fontweight='600')
    ax.set_title(subtitle, fontsize=10, fontweight='600')
    ax.legend(fontsize=8, loc="best", frameon=True, shadow=True)
    if max_months:
        ax.set_xlim(0, max_months)
    else:
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

    # Add horizontal padding so rightmost year/value labels do not touch the canvas edge.
    ax.set_xlim(window_start - 0.4, window_end + 0.6)
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
    fig.text(0.01, 0.015, " ".join(note_parts), fontsize=7, ha="left", color='#718096')
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.92)
    fig.savefig(outpath, dpi=300, facecolor='white')
    plt.close(fig)

def save_bar_cantones(cant: pd.DataFrame, outpath: str, title: str, geo_label: str = "canton"):
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

        geo_label_norm = (geo_label or "canton").strip().lower()
        if geo_label_norm == "parroquia":
            note_parts = ["Unidad: RUC por parroquia principal."]
        else:
            note_parts = ["Unidad: RUC por cant?n principal."]
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
    # Excluir filas fuera de ventana (etiquetas tipo "FUERA DE VENTANA")
    cohort_upper = df["cohort_5y"].str.upper().fillna("")
    mask_out = cohort_upper.str.contains("FUERA") & cohort_upper.str.contains("VENTANA")
    births_col = "births_n" if "births_n" in df.columns else None
    closures_col = "closures_terminal_n" if "closures_terminal_n" in df.columns else None
    if births_col is None and closures_col is None:
        ax.text(0.5, 0.5, "Sin nacimientos/cierres en cohortes", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return

    excluded = df[mask_out].copy()
    df = df[~mask_out].copy()
    if df.empty:
        ax.text(0.5, 0.5, "Sin cohortes en ventana", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return
    df = df.sort_values("cohort_5y").reset_index(drop=True)
    positions = range(len(df))
    width = 0.38
    births_total = 0.0
    closures_total = 0.0
    if births_col:
        df[births_col] = pd.to_numeric(df[births_col], errors="coerce").fillna(0)
        births_total = float(df[births_col].sum())
        births_bars = ax.bar(
            [p - width / 2 for p in positions],
            df[births_col],
            width=width,
            label=f"Nacimientos (n={_fmt_int(births_total)})",
            color=COLOR_PALETTE[0],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )
    if closures_col:
        df[closures_col] = pd.to_numeric(df[closures_col], errors="coerce").fillna(0)
        closures_total = float(df[closures_col].sum())
        closures_bars = ax.bar(
            [p + width / 2 for p in positions],
            df[closures_col],
            width=width,
            label=f"Cierres (n={_fmt_int(closures_total)})",
            color="#e53e3e",
            edgecolor="white",
            linewidth=1.2,
            alpha=0.9,
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(df["cohort_5y"], rotation=45, ha="right")
    ax.set_ylabel("Conteo (RUC)")
    ymax = max(df[births_col].max() if births_col else 0, df[closures_col].max() if closures_col else 0)
    ymax = max(float(ymax), 1.0)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_title(title)
    ax.legend(fontsize=8)

    def _annotate_bars(bars, total):
        if total <= 0:
            return
        for bar in bars:
            value = float(bar.get_height())
            if value <= 0:
                continue
            pct = value / total * 100
            label = f"{int(round(value)):,} ({pct:.1f}%)"
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(
                x,
                y + ymax * 0.03,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="600",
                color="#2d3748",
            )

    if births_col:
        _annotate_bars(births_bars, births_total)
    if closures_col:
        _annotate_bars(closures_bars, closures_total)

    note_parts = [
        "Cohortes quinquenales por inicio de actividades.",
        f"Nacimientos total: {_fmt_int(births_total)}.",
        f"Cierres total: {_fmt_int(closures_total)}.",
    ]
    if births_total > 0 and closures_total > 0:
        note_parts.append(f"Cierres/Nacimientos: {_fmt_percent(closures_total / births_total)}.")
    if not excluded.empty:
        excl_births = float(pd.to_numeric(excluded.get(births_col), errors="coerce").fillna(0).sum()) if births_col else 0.0
        excl_closures = float(pd.to_numeric(excluded.get(closures_col), errors="coerce").fillna(0).sum()) if closures_col else 0.0
        note_parts.append(
            f"Excluido fuera de ventana: Nacimientos={_fmt_int(excl_births)}, Cierres={_fmt_int(excl_closures)}."
        )
    fig.text(0.01, 0.01, " ".join(note_parts), fontsize=7, ha="left", color="#718096")
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


def save_km_multi(km_map: dict[str, pd.DataFrame], outpath: str, title: str,
                  max_months: float | None = None, top_n: int | None = None,
                  label_prefix: str = "", fill: bool = True,
                  fill_alpha: float = 0.15,
                  group_counts: dict[str, int] | None = None,
                  group_stats: dict[str, dict[str, int]] | None = None,
                  label_map: dict[str, str] | None = None,
                  show_last_point: bool = True,
                  milestone_times: list[int] | None = None,
                  milestone_label_time: int | None = None,
                  line_styles: list[str] | None = None,
                  legend_only: list[str] | None = None,
                  legend_only_suffix: str | None = None,
                  at_risk: dict[str, list[int]] | None = None,
                  at_risk_times: list[int] | None = None,
                  extra_note: str | None = None):
    # Filtrar a los top_n grupos por longitud de curva KM
    if top_n and km_map:
        sorted_keys = sorted(km_map.keys(),
                             key=lambda k: len(km_map[k]) if km_map[k] is not None else 0,
                             reverse=True)[:top_n]
        km_map = {k: km_map[k] for k in sorted_keys}
    total_shown = len(km_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    if not km_map:
        ax.text(0.5, 0.5, "KM estratificado no disponible (umbral insuficiente)", ha="center", va="center")
    else:
        _plot_km_multi(
            ax,
            km_map,
            "",
            max_months=max_months,
            label_prefix=label_prefix,
            fill=fill,
            fill_alpha=fill_alpha,
            group_counts=group_counts,
            group_stats=group_stats,
            label_map=label_map,
            show_last_point=show_last_point,
            milestone_times=milestone_times,
            milestone_label_time=milestone_label_time,
            line_styles=line_styles,
            legend_only=legend_only,
            legend_only_suffix=legend_only_suffix,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.set_title(title, fontweight='bold', color='#2d3748')
    note = "Asociativo, no causal."
    if top_n:
        note += f"  Se muestran hasta {top_n} grupos con más observaciones."
    if extra_note:
        note += f"  {extra_note}"
    note_y = 0.01
    bottom_pad = 0.04
    if at_risk and at_risk_times:
        _add_at_risk_table(ax, at_risk_times, at_risk, label_map=label_map, row_order=list(km_map.keys()))
        ax.tick_params(axis='x', pad=4)
        bottom_pad = 0.22 + 0.06 * len(at_risk)
        note_y = min(0.12 + 0.02 * len(at_risk), 0.22)
    fig.text(0.01, note_y, note, fontsize=8, ha="left", color='#718096')
    fig.tight_layout(rect=[0, bottom_pad, 1, 1])
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

    def _normalize_flag_label(value: str) -> str:
        v = str(value).strip().upper()
        if v in {"SI", "SÍ"}:
            return "Sí"
        if v == "NO":
            return "No"
        if v in {"NO INFORMADO", "N/I", "NA"}:
            return "No informado"
        return str(value)

    rows = len(km_flags)
    fig, axes = plt.subplots(rows, 1, figsize=(8, 3.5 * rows))
    if rows == 1:
        axes = [axes]

    title_map = {
        "obligado_3cat": "Kaplan–Meier por Obligación (Sí vs No)",
        "agente_retencion_3cat": "Kaplan–Meier por Agente de Retención (Sí vs No)",
        "especial_3cat": "Kaplan–Meier por Contribuyente Especial (Sí vs No)",
    }

    for ax, (flag, km_map) in zip(axes, km_flags.items(), strict=False):
        group_counts: dict[str, int] = {}
        label_map: dict[str, str] = {}
        for grp, km in km_map.items():
            label_map[grp] = _normalize_flag_label(grp)
            if km is None or km.empty or "n_at_risk" not in km.columns:
                continue
            try:
                n_val = int(pd.to_numeric(km["n_at_risk"], errors="coerce").max())
            except (TypeError, ValueError):
                continue
            group_counts[grp] = n_val

        label_map.setdefault("SI", "Sí")
        label_map.setdefault("SÍ", "Sí")
        label_map.setdefault("NO", "No")

        has_yes = any(_normalize_flag_label(g) == "Sí" for g in km_map.keys())
        has_no = any(_normalize_flag_label(g) == "No" for g in km_map.keys())
        legend_only: list[str] = []
        if not has_yes:
            legend_only.append("SI")
            group_counts.setdefault("SI", 0)
        if not has_no:
            legend_only.append("NO")
            group_counts.setdefault("NO", 0)

        max_months = 300
        _plot_km_multi(
            ax,
            km_map,
            title_map.get(flag, f"Kaplan–Meier por {flag}"),
            max_months=max_months,
            fill=False,
            group_counts=group_counts if group_counts else None,
            label_map=label_map if label_map else None,
            legend_only=legend_only if legend_only else None,
            legend_only_suffix="sin curva",
        )

        t_ref = 60 if max_months >= 60 else 120 if max_months >= 120 else None
        if t_ref and km_map:
            parts = []
            for grp, km in km_map.items():
                s_val = _survival_at_time(km, t_ref)
                lbl = label_map.get(grp, grp) if label_map else grp
                parts.append(f"{lbl}={_fmt_percent(s_val)}")
            note = f"S({t_ref}m): " + ", ".join(parts)
            ax.text(
                0.01,
                0.02,
                note,
                transform=ax.transAxes,
                fontsize=7,
                color="#4a5568",
                ha="left",
                va="bottom",
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Asociativo, no causal.", fontsize=7, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_heatmap_placeholder(outpath: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "Heatmap geográfico no disponible", ha="center", va="center")
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
    geo_level: str = "canton",
    canton: str | None = None,
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

    join_col = "parroquia" if (geo_level or "").strip().lower() == "parroquia" else "canton"
    if join_col not in df.columns:
        save_heatmap_placeholder(outpath, title)
        return

    value_col = "ruc_share" if "ruc_share" in df.columns else "establishments_share"
    level_label = "parroquial" if join_col == "parroquia" else "cantonal"
    value_label = (
        f"Participacion RUC ({level_label})"
        if value_col == "ruc_share"
        else f"Participacion establecimientos ({level_label})"
    )

    df = df.copy()
    # Usar operaciones vectorizadas
    df["_join"] = (
        df[join_col]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.normalize('NFKD')
        .str.encode('ascii', 'ignore')
        .str.decode('ascii')
    )

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception:
        save_heatmap_placeholder(outpath, title)
        return

    name_col = None
    if join_col == "parroquia":
        name_candidates = ["DPA_DESPAR", "DPA_DESPAR_", "DPA_DESPARM", "DPA_DES_PAR", "PARROQUIA"]
    else:
        name_candidates = ["DPA_DESCAN", "DPA_DESCAN_", "DPA_DESCANM", "DPA_DES_CAN", "CANTON"]

    for col in name_candidates:
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

    if join_col == "parroquia" and canton and "DPA_DESCAN" in gdf.columns:
        canton_norm = _normalize_geo_name(canton)
        gdf_canton = gdf["DPA_DESCAN"].str.strip().str.upper()
        gdf_canton = gdf_canton.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii')
        gdf_canton = gdf_canton.str.split().str.join(' ')
        gdf = gdf[gdf_canton == canton_norm].copy()

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
        (
            "RUC multi-provincia",
            _fmt_share_num(
                kpis.get("multi_province_n"),
                kpis.get("multi_province_total"),
                kpis.get("multi_province_share"),
            ),
        ),
        ("% registros con faltantes críticos", _fmt_percent(kpis.get("missing_critical_any_share"))),
        ("Top 3 faltantes (cols)", kpis.get("missing_critical_top3_cols") or "N/A"),
        ("Faltantes críticos (promedio)", _fmt_percent(kpis.get("missing_critical_avg_share"))),
        ("Faltantes críticos (máx)", _fmt_percent(kpis.get("missing_critical_max_share"))),
    ]

    demo_rows = [
        ("Nacimientos 2000-2024", _fmt_num_share(kpis.get("births_total_2000_2024"), kpis.get("births_share_universe"))),
        ("Cierres 2000-2024", _fmt_num_share(kpis.get("closures_terminal_total_2000_2024"), kpis.get("closures_share_universe"))),
        ("Neto 2000-2024", _fmt_num_share(kpis.get("net_total_2000_2024"), kpis.get("net_share_universe"))),
        ("Tasa nacimientos anual prom.", _fmt_rate(kpis.get("birth_rate_avg"))),
        ("Tasa cierres anual prom.", _fmt_rate(kpis.get("closure_rate_avg"))),
        ("Tasa neta anual prom.", _fmt_rate(kpis.get("net_rate_avg"))),
        ("Nacimientos últimos 5 años", _fmt_int(kpis.get("births_last5"))),
        ("Cierres últimos 5 años", _fmt_int(kpis.get("closures_last5"))),
        ("Neto últimos 5 años", _fmt_int(kpis.get("net_last5"))),
        ("Tasa nacimientos anual prom. (ult. 5a)", _fmt_rate(kpis.get("birth_rate_last5"))),
        ("Tasa cierres anual prom. (ult. 5a)", _fmt_rate(kpis.get("closure_rate_last5"))),
        ("Tasa neta anual prom. (ult. 5a)", _fmt_rate(kpis.get("net_rate_last5"))),
    ]

    struct_rows = [
        ("Top3 cantonal (RUC)", _fmt_percent(kpis.get("top3_concentration_by_ruc_share"))),
        ("Top5 cantonal (RUC)", _fmt_percent(kpis.get("top5_concentration_by_ruc_share"))),
        ("Top3 cantonal (estab)", _fmt_percent(kpis.get("top3_concentration_by_establishments_share"))),
        ("Top5 cantonal (estab)", _fmt_percent(kpis.get("top5_concentration_by_establishments_share"))),
        ("Top 3 cantones (RUC)", kpis.get("top3_cantons") or "N/A"),
        (
            "Cantón líder",
            f"{kpis.get('leading_canton', 'N/A')} ({_fmt_percent(kpis.get('leading_canton_share'))})",
        ),
        ("Top 3 macro-sectores", kpis.get("top3_macro_sectors") or "N/A"),
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
        ("S(12m)", _fmt_percent(kpis.get("S_12m"))),
        ("S(24m)", _fmt_percent(kpis.get("S_24m"))),
        ("S(60m)", _fmt_percent(kpis.get("S_60m"))),
        ("S(120m)", _fmt_percent(kpis.get("S_120m"))),
        ("S(300m)", _fmt_percent(kpis.get("S_300m"))),
        ("Mediana supervivencia", _fmt_months(kpis.get("median_survival_months"))),
        ("Cierre temprano (<12m)", _fmt_percent(kpis.get("early_closure_share_lt_12m"))),
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

    fig_height = max(12, 0.32 * len(rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
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
    defs = [
        f"Periodo {window_start}-{window_end} · Censura {censor_date} (administrativa al final del periodo). Asociativo, no causal.",
        "Definiciones:",
        "- Filas raw: registros de establecimientos en el raw filtrado (no anualizado).",
        "- Faltantes criticos: CODIGO_CIIU, FECHA_INICIO_ACTIVIDADES, FECHA_SUSPENSION_DEFINITIVA, OBLIGADO, AGENTE_RETENCION, ESPECIAL.",
        "- RUC multi-provincia: RUC con establecimientos en >=2 provincias.",
        "- Nacimientos = primera fecha de inicio; cierres = suspension definitiva/terminal; periodo critico = intervalo con mayor densidad de cierres (bins configurados).",
    ]
    fig.text(0.01, 0.01, "\n".join(defs), fontsize=7, ha="left", color='#718096')
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_report_flow(outpath: str, title: str, steps: list[str] | None = None) -> None:
    flow_steps = steps or [
        "Resumen ejecutivo (10 KPIs)",
        "Demografía",
        "Estructura sectorial",
        "Supervivencia global",
        "Supervivencia segmentada",
        "Anexos técnicos",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if not flow_steps:
        ax.text(0.5, 0.5, "Sin pasos definidos", ha="center", va="center", fontsize=12)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    n_steps = len(flow_steps)
    y_top = 0.88
    y_bottom = 0.12
    spacing = (y_top - y_bottom) / max(n_steps - 1, 1)
    box_height = min(0.11, spacing * 0.7)
    box_width = 0.78
    x_left = (1 - box_width) / 2

    for idx, label in enumerate(flow_steps):
        y_center = y_top - idx * spacing
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        box = FancyBboxPatch(
            (x_left, y_center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=1.2,
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(
            0.5,
            y_center,
            f"{idx + 1}. {label}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        if idx < n_steps - 1:
            y_next = y_top - (idx + 1) * spacing
            ax.annotate(
                "",
                xy=(0.5, y_next + box_height / 2 + 0.012),
                xytext=(0.5, y_center - box_height / 2 - 0.012),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.6),
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#2d3748")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_executive_kpi_dashboard(
    kpis: dict[str, object],
    outpath: str,
    title: str,
) -> None:
    kpi_items = [
        ("RUC unicos", _fmt_int(kpis.get("unique_ruc_in_province")), "Universo"),
        ("Filas raw", _fmt_int(kpis.get("raw_rows_establishments")), "Universo"),
        ("Estab/RUC mediana", _fmt_float(kpis.get("establishments_per_ruc_median")), "Universo"),
        ("Nacimientos 2000-2024", _fmt_int(kpis.get("births_total_2000_2024")), "Demografia"),
        ("Cierres 2000-2024", _fmt_int(kpis.get("closures_terminal_total_2000_2024")), "Demografia"),
        ("Neto 2000-2024", _fmt_int(kpis.get("net_total_2000_2024")), "Demografia"),
        ("Supervivencia a 24 meses", _fmt_percent(kpis.get("S_24m")), "Supervivencia"),
        ("Supervivencia a 60 meses", _fmt_percent(kpis.get("S_60m")), "Supervivencia"),
        ("Mediana supervivencia", _fmt_months(kpis.get("median_survival_months")), "Supervivencia"),
        ("Cierre temprano <24m", _fmt_percent(kpis.get("early_closure_share_lt_24m")), "Supervivencia"),
    ]

    category_colors = {
        "Universo": "#2563eb",
        "Demografia": "#14b8a6",
        "Supervivencia": "#f97316",
    }

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#f7f8fc")
    ax.set_facecolor("#f7f8fc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        title,
        fontsize=16,
        fontweight="bold",
        color="#1f2937",
        ha="left",
        va="top",
    )
    window_start = kpis.get("window_start_year")
    window_end = kpis.get("window_end_year")
    censor_date = kpis.get("censor_date")
    subtitle = f"Periodo {window_start}-{window_end} · Censura {censor_date}"
    ax.text(0.02, 0.91, subtitle, fontsize=9.5, color="#6b7280", ha="left", va="top")
    cols = 5
    rows = 2
    pad = 0.02
    top_margin = 0.18
    card_w = (1 - pad * (cols + 1)) / cols
    card_h = (1 - top_margin - pad * (rows + 1)) / rows
    top_y = 1 - top_margin - pad - card_h

    for idx, (label, value, category) in enumerate(kpi_items):
        row = idx // cols
        col = idx % cols
        x = pad + col * (card_w + pad)
        y = top_y - row * (card_h + pad)
        color = category_colors.get(category, "#4a5568")
        shadow = FancyBboxPatch(
            (x + 0.006, y - 0.008),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#dbe0ea",
            edgecolor="none",
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(shadow)

        card = FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor="#e5e7eb",
            linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(card)

        bar = FancyBboxPatch(
            (x, y + card_h - 0.015),
            card_w,
            0.015,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(bar)

        ax.text(
            x + 0.04 * card_w,
            y + 0.72 * card_h,
            label,
            fontsize=9.5,
            fontweight="600",
            color="#4b5563",
            ha="left",
            va="center",
            zorder=4,
        )
        ax.text(
            x + 0.04 * card_w,
            y + 0.38 * card_h,
            value,
            fontsize=13.5,
            fontweight="bold",
            color="#111827",
            ha="left",
            va="center",
            zorder=4,
        )
        ax.text(
            x + 0.04 * card_w,
            y + 0.15 * card_h,
            category,
            fontsize=8.5,
            color=color,
            ha="left",
            va="center",
            zorder=4,
        )

    note = "Asociativo, no causal."
    fig.text(0.01, 0.01, note, fontsize=8, ha="left", color="#718096")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="#f7f8fc")
    plt.close(fig)


def save_qc_dashboard(qc_raw: dict, qc_ruc: dict, outpath: str, title: str, qc_extra: dict | None = None) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1, 1], hspace=0.35, wspace=0.25)
    sem_ax = fig.add_subplot(gs[0, :])
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    # Agregar título principal con estilo
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#2d3748', y=0.97)

    raw_rows = _fmt_int(qc_raw.get("raw_rows"))
    unique_ruc = _fmt_int(qc_raw.get("unique_ruc"))
    raw_rows_table = [
        ("Filas raw", raw_rows),
        ("RUC únicos", unique_ruc),
    ]

    domains = qc_raw.get("domains", {}) or {}
    domain_labels = {
        "CLASE_CONTRIBUYENTE_in_GEN_RMP_SIM_share": "Clase contribuyente (GEN/RMP/SIM)",
        "ESTADO_CONTRIBUYENTE_in_ACTIVO_PASIVO_SUSPENDIDO_share": "Estado contribuyente (Activo/Pasivo/Suspendido)",
        "ESTADO_ESTABLECIMIENTO_in_ABI_CER_share": "Estado establecimiento (ABI/CER)",
        "TIPO_CONTRIBUYENTE_in_PERSONA_SOCIEDAD_share": "Tipo contribuyente (Persona/Sociedad)",
        "CODIGO_JURISDICCION_non_empty_share": "Codigo jurisdiccion no vacio",
    }
    domains_rows = [(label, _fmt_percent(domains.get(key))) for key, label in domain_labels.items()]

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
    missing_rows = [(_pretty_label(col), _fmt_percent(val)) for col, val in top_missing]
    if not missing_rows:
        missing_rows = [("Sin datos", "N/A")]

    # Semáforo QC
    sem_ax.set_axis_off()
    sem_ax.set_xlim(0, 1)
    sem_ax.set_ylim(0, 1)
    sem_ax.text(0.01, 0.92, "Semáforo QC", fontsize=11, fontweight="bold", color="#2d3748", ha="left")

    domains = qc_raw.get("domains", {}) or {}
    domain_keys = [
        "CLASE_CONTRIBUYENTE_in_GEN_RMP_SIM_share",
        "ESTADO_CONTRIBUYENTE_in_ACTIVO_PASIVO_SUSPENDIDO_share",
        "ESTADO_ESTABLECIMIENTO_in_ABI_CER_share",
        "TIPO_CONTRIBUYENTE_in_PERSONA_SOCIEDAD_share",
        "CODIGO_JURISDICCION_non_empty_share",
    ]
    domain_vals = [
        float(domains.get(k))
        for k in domain_keys
        if isinstance(domains.get(k), (int, float)) and math.isfinite(domains.get(k))
    ]
    domains_min = min(domain_vals) if domain_vals else float("nan")

    suspendido_sin_fecha = int((qc_ruc.get("state_vs_dates_audit", {}) or {}).get("suspendido_without_suspension_date_n") or 0)
    activo_con_cierre = int((qc_ruc.get("state_vs_dates_audit", {}) or {}).get("activo_with_terminal_suspension_n") or 0)
    negative_n = int(qc_ruc.get("negative_durations_n") or 0)
    ruc_rows = int(qc_ruc.get("ruc_rows") or 0)
    negative_share = (negative_n / ruc_rows) if ruc_rows else float("nan")

    multi_share = float("nan")
    if qc_extra:
        multi_share = float((qc_extra.get("multi_province", {}) or {}).get("ruc_multi_province_share", float("nan")))

    def _status_color(kind: str, value: float | int | None) -> str:
        if value is None or not math.isfinite(float(value)):
            return "#9ca3af"
        v = float(value)
        if kind == "domains":
            if v >= 0.99:
                return "#22c55e"
            if v >= 0.95:
                return "#f59e0b"
            return "#ef4444"
        if kind == "suspendido":
            if v == 0:
                return "#22c55e"
            if v <= 10:
                return "#f59e0b"
            return "#ef4444"
        if kind == "activo_cierre":
            if v <= 5:
                return "#22c55e"
            if v <= 20:
                return "#f59e0b"
            return "#ef4444"
        if kind == "neg_share":
            if v <= 0.01:
                return "#22c55e"
            if v <= 0.05:
                return "#f59e0b"
            return "#ef4444"
        if kind == "multi_share":
            if v <= 0.1:
                return "#22c55e"
            if v <= 0.25:
                return "#f59e0b"
            return "#ef4444"
        return "#9ca3af"

    sem_items = [
        ("Dominios esperados (mín.)", _fmt_percent(domains_min), _status_color("domains", domains_min)),
        ("Suspendido sin fecha", _fmt_int(suspendido_sin_fecha), _status_color("suspendido", suspendido_sin_fecha)),
        ("Activo con cierre", _fmt_int(activo_con_cierre), _status_color("activo_cierre", activo_con_cierre)),
        ("Duraciones negativas", _fmt_percent(negative_share), _status_color("neg_share", negative_share)),
        ("Multi-prov", _fmt_percent(multi_share), _status_color("multi_share", multi_share)),
    ]
    n_items = len(sem_items)
    for idx, (label, value, color) in enumerate(sem_items):
        x = (idx + 0.5) / n_items
        sem_ax.add_patch(
            mpl.patches.Circle((x, 0.58), 0.035, color=color, alpha=0.95, transform=sem_ax.transAxes)
        )
        sem_ax.text(x, 0.72, label, ha="center", va="center", fontsize=9, color="#4b5563")
        sem_ax.text(x, 0.48, value, ha="center", va="center", fontsize=10, fontweight="bold", color="#1f2937")

    _render_table(axes[0], raw_rows_table, "QC base (raw)")
    _render_table(axes[1], domains_rows, "Dominios esperados")
    if extra_rows:
        _render_table(axes[2], ruc_rows_table + extra_rows, "QC RUC + contexto")
    else:
        _render_table(axes[2], ruc_rows_table, "QC RUC")
    _render_table(axes[3], missing_rows, "Top faltantes")

    # `tight_layout` emite warning con esta combinación de tablas/axes.
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.93, wspace=0.25, hspace=0.35)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def save_metrics_dashboard(metrics: dict, outpath: str, title: str) -> None:
    run = metrics.get("run", {}) or {}
    dem = metrics.get("demography", {}) or {}
    geo = metrics.get("geography", {}) or {}
    sector = metrics.get("sector", {}) or {}
    surv = metrics.get("survival", {}) or {}
    def _as_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _ratio_phrase(pct: float | None) -> str:
        if pct is None or not math.isfinite(pct):
            return "N/A"
        p = float(pct)
        if p >= 0.875:
            return "≈ 9 de cada 10"
        if p >= 0.8:
            return "≈ 4 de cada 5"
        if p >= 0.75:
            return "≈ 3 de cada 4"
        if p >= 0.67:
            return "≈ 2 de cada 3"
        if p >= 0.6:
            return "≈ 3 de cada 5"
        if p >= 0.5:
            return "≈ 1 de cada 2"
        if p > 0:
            denom = max(int(round(1 / p)), 2)
            return f"≈ 1 de cada {denom}"
        return "N/A"

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f7f8fc")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.35, wspace=0.28)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_run = fig.add_subplot(gs[1, 0])
    ax_geo = fig.add_subplot(gs[1, 1])
    ax_sector = fig.add_subplot(gs[2, 0])
    ax_surv = fig.add_subplot(gs[2, 1])

    for ax in [ax_cards, ax_run, ax_geo, ax_sector, ax_surv]:
        ax.set_facecolor("#f7f8fc")
        ax.set_axis_off()

    # KPI cards
    births = dem.get("births_total_2000_2024")
    closures = dem.get("closures_terminal_total_2000_2024")
    net = dem.get("net_total_2000_2024")
    s60 = surv.get("S_60m")
    s120 = surv.get("S_120m")
    median = surv.get("median_survival_months")
    early24 = surv.get("early_closure_share_lt_24m")

    cards = [
        ("Nacimientos", _fmt_int(births), "#2563eb", None),
        ("Cierres", _fmt_int(closures), "#ef4444", None),
        ("Neto", _fmt_int(net), "#14b8a6", None),
        ("Mediana supervivencia", _fmt_months(median), "#f59e0b", None),
        ("S(60m)", _fmt_percent(s60), "#6366f1", f"S(120m) {_fmt_percent(s120)}"),
        ("% cierres <24m", _fmt_percent(early24), "#0ea5e9", None),
    ]

    ax_cards.text(
        0.02,
        0.96,
        title,
        fontsize=16,
        fontweight="bold",
        color="#1f2937",
        ha="left",
        va="top",
    )
    subtitle = f"Provincia {run.get('province', 'N/A')} · Ventana {run.get('window_start_year', 'N/A')}-{run.get('window_end_year', 'N/A')}"
    ax_cards.text(0.02, 0.86, subtitle, fontsize=9.5, color="#6b7280", ha="left", va="top")

    cols = 3
    rows = 2
    pad = 0.03
    top_margin = 0.28
    card_w = (1 - pad * (cols + 1)) / cols
    card_h = (1 - top_margin - pad * (rows + 1)) / rows
    top_y = 1 - top_margin - pad - card_h

    for idx, (label, value, accent, subline) in enumerate(cards):
        row = idx // cols
        col = idx % cols
        x = pad + col * (card_w + pad)
        y = top_y - row * (card_h + pad)
        shadow = FancyBboxPatch(
            (x + 0.006, y - 0.008),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#dbe0ea",
            edgecolor="none",
            alpha=0.35,
            zorder=1,
        )
        ax_cards.add_patch(shadow)
        card = FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor="#e5e7eb",
            linewidth=1.0,
            zorder=2,
        )
        ax_cards.add_patch(card)
        bar = FancyBboxPatch(
            (x, y + card_h - 0.015),
            card_w,
            0.015,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor=accent,
            edgecolor="none",
            zorder=3,
        )
        ax_cards.add_patch(bar)
        ax_cards.text(
            x + 0.04 * card_w,
            y + 0.72 * card_h,
            label,
            fontsize=9.5,
            fontweight="600",
            color="#4b5563",
            ha="left",
            va="center",
            zorder=4,
        )
        ax_cards.text(
            x + 0.96 * card_w,
            y + 0.38 * card_h,
            value,
            fontsize=13.5,
            fontweight="bold",
            color="#111827",
            ha="right",
            va="center",
            zorder=4,
        )
        if subline:
            ax_cards.text(
                x + 0.96 * card_w,
                y + 0.16 * card_h,
                subline,
                fontsize=8.5,
                color="#6b7280",
                ha="right",
                va="center",
                zorder=4,
            )

    # Run block
    ax_run.set_axis_off()
    ax_run.text(0.02, 0.92, "Run", fontsize=11, fontweight="bold", color="#2d3748", ha="left")
    run_items = [
        ("Ventana", f"{run.get('window_start_year', 'N/A')} - {run.get('window_end_year', 'N/A')}"),
        ("Censura", str(run.get("censor_date", "N/A"))),
        ("RUC incluidos", _fmt_int((metrics.get("survival", {}) or {}).get("km", {}).get("n_total"))),
    ]
    y = 0.74
    for label, value in run_items:
        ax_run.text(0.02, y, label, fontsize=9, color="#4b5563", ha="left")
        ax_run.text(0.98, y, value, fontsize=9.5, color="#111827", ha="right")
        y -= 0.18

    # Geografia block
    ax_geo.set_axis_off()
    ax_geo.text(0.02, 0.92, "Geografía", fontsize=11, fontweight="bold", color="#2d3748", ha="left")
    geo_vals = [
        ("RUC Top3", _as_float(geo.get("top3_concentration_by_ruc_share"))),
        ("RUC Top5", _as_float(geo.get("top5_concentration_by_ruc_share"))),
        ("Estab Top3", _as_float(geo.get("top3_concentration_by_establishments_share"))),
        ("Estab Top5", _as_float(geo.get("top5_concentration_by_establishments_share"))),
    ]
    bars_ax = ax_geo.inset_axes([0.05, 0.18, 0.9, 0.62])
    bars_ax.set_facecolor("#f7f8fc")
    bars_ax.set_xlim(0, 1.0)
    bars_ax.set_ylim(-0.5, len(geo_vals) - 0.5)
    bars_ax.spines["top"].set_visible(False)
    bars_ax.spines["right"].set_visible(False)
    bars_ax.spines["left"].set_visible(False)
    bars_ax.spines["bottom"].set_visible(False)
    bars_ax.set_xticks([])
    bars_ax.set_yticks(range(len(geo_vals)))
    bars_ax.set_yticklabels([g[0] for g in geo_vals], fontsize=8.5)
    colors = ["#2563eb", "#2563eb", "#14b8a6", "#14b8a6"]
    for idx, (label, val) in enumerate(geo_vals):
        v = val if math.isfinite(val) else 0.0
        bars_ax.barh(idx, v, color=colors[idx], alpha=0.85, height=0.5)
        txt = _fmt_percent(val) if math.isfinite(val) else "N/A"
        bars_ax.text(1.02, idx, txt, va="center", ha="left", fontsize=8, color="#4b5563")
    bars_ax.invert_yaxis()

    top5_ruc = _as_float(geo.get("top5_concentration_by_ruc_share"))
    if math.isfinite(top5_ruc):
        if top5_ruc >= 0.9:
            conc_label = "altísima concentración territorial"
        elif top5_ruc >= 0.75:
            conc_label = "alta concentración territorial"
        elif top5_ruc >= 0.6:
            conc_label = "concentración moderada"
        else:
            conc_label = "baja concentración"
        geo_line = f"Top5={_fmt_percent(top5_ruc)} ({conc_label})"
    else:
        geo_line = "Top5=N/A"
    ax_geo.text(0.02, 0.08, geo_line, fontsize=8, color="#4b5563", ha="left")

    # Sector block
    ax_sector.set_axis_off()
    ax_sector.text(0.02, 0.92, "Sector", fontsize=11, fontweight="bold", color="#2d3748", ha="left")
    leading = sector.get("leading_macro_sector", {}) or {}
    div = sector.get("diversification", {}) or {}
    lead_letter = str(leading.get("letter", "N/A"))
    lead_label_full = str(leading.get("label", "N/A"))
    lead_label = _trim_text(lead_label_full, max_len=26)
    lead_share = _fmt_percent(leading.get("share"))
    hhi = _as_float(div.get("hhi_macro_sector"))
    eff = _as_float(div.get("effective_macro_sectors"))

    if math.isfinite(hhi):
        if hhi < 0.15:
            hhi_txt = "HHI bajo → diversificado"
            hhi_color = "#22c55e"
        elif hhi < 0.25:
            hhi_txt = "HHI medio → concentración moderada"
            hhi_color = "#f59e0b"
        else:
            hhi_txt = "HHI alto → concentrado"
            hhi_color = "#ef4444"
    else:
        hhi_txt = "HHI N/A"
        hhi_color = "#9ca3af"

    ax_sector.text(0.02, 0.7, f"Macro líder: {lead_letter}", fontsize=9, color="#4b5563", ha="left")
    ax_sector.text(0.98, 0.7, lead_share, fontsize=9.5, color="#111827", ha="right")
    ax_sector.text(0.02, 0.54, f"Etiqueta: {lead_label}", fontsize=9, color="#4b5563", ha="left")
    ax_sector.text(0.02, 0.34, f"HHI={_fmt_float(hhi)}", fontsize=9, color="#4b5563", ha="left")
    ax_sector.text(0.98, 0.34, hhi_txt, fontsize=9, color=hhi_color, ha="right")
    ax_sector.text(
        0.02,
        0.14,
        f"Macros efectivos: {_fmt_float(eff)} (≈ N sectores equilibrados)",
        fontsize=8.5,
        color="#4b5563",
        ha="left",
    )

    # Supervivencia block
    ax_surv.set_axis_off()
    ax_surv.text(0.02, 0.92, "Supervivencia", fontsize=11, fontweight="bold", color="#2d3748", ha="left")
    surv_items = [
        ("Supervivencia 60m", _fmt_percent(s60)),
        ("Supervivencia 120m", _fmt_percent(s120)),
        ("Mediana", _fmt_months(median)),
        ("% cierres <24m", _fmt_percent(early24)),
    ]
    y = 0.74
    for label, value in surv_items:
        ax_surv.text(0.02, y, label, fontsize=9, color="#4b5563", ha="left")
        ax_surv.text(0.98, y, value, fontsize=9.5, color="#111827", ha="right")
        y -= 0.16

    s60_val = _as_float(s60)
    if math.isfinite(s60_val):
        surv_line = f"Supervivencia 60m={_fmt_percent(s60_val)} ({_ratio_phrase(s60_val)})"
    else:
        surv_line = "S(60m)=N/A"
    ax_surv.text(0.02, 0.08, surv_line, fontsize=8, color="#4b5563", ha="left")

    raw_filename = str((run.get("inputs", {}) or {}).get("raw_filename", "N/A"))
    footer_parts = [f"Raw: {raw_filename}"]
    if lead_label_full and lead_label_full != lead_label:
        footer_parts.append(f"Macro líder (completo): {lead_label_full}")
    footer = " · ".join(footer_parts)
    fig.text(0.01, 0.01, footer, fontsize=7, ha="left", color="#718096")
    # `tight_layout` no es compatible con todos los ejes de este dashboard.
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.93, wspace=0.28, hspace=0.35)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="#f7f8fc", edgecolor="none")
    plt.close(fig)
