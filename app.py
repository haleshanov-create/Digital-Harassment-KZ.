import os
import re
import pandas as pd
import gradio as gr
import plotly.express as px

# ---------- НАСТРОЙКИ ----------
CSV_PATH = os.environ.get(
    "DATASET_PATH",
    "Digital_Harassment_KZ_dataset.csv"
)

BG = "#050914"
PANEL = "#0B1224"
CYAN = "#22D3EE"
PURPLE = "#A78BFA"
TEXT = "#E5F3FF"


# ---------- ЗАГРУЗКА ДАННЫХ ----------
def load_dataset():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        try:
            return pd.read_csv(CSV_PATH, sep=";")
        except Exception:
            return pd.DataFrame()


df = load_dataset()


# Распознаём возможные варианты названий колонок
def find_column(frame, candidates):
    lookup = {str(c).strip().lower(): c for c in frame.columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


TITLE_COL = find_column(
    df, ["title", "заголовок", "название"]
)
DATE_COL = find_column(
    df, ["date", "published", "publication_date", "дата"]
)
SUMMARY_COL = find_column(
    df, ["summary", "description", "описание", "краткое описание"]
)
CATEGORY_COL = find_column(
    df, ["category", "категория", "categories"]
)
SOURCE_COL = find_column(
    df, ["source", "источник", "domain"]
)
URL_COL = find_column(
    df, ["url", "link", "ссылка"]
)


def get_series(column, default="Не указано"):
    if column is None:
        return pd.Series(default, index=df.index, dtype="object")

    return df[column].fillna(default).astype(str).str.strip()


if not df.empty:
    df["_title"] = get_series(TITLE_COL, "Без заголовка")
    df["_date"] = get_series(DATE_COL)
    df["_summary"] = get_series(SUMMARY_COL, "")
    df["_category"] = get_series(CATEGORY_COL)
    df["_source"] = get_series(SOURCE_COL)
    df["_url"] = get_series(URL_COL, "")

    df["_date_parsed"] = pd.to_datetime(
        df["_date"], errors="coerce", utc=True
    )


# ---------- СТИЛИ ----------
CSS = """
.gradio-container {
    background:
        radial-gradient(ellipse at top left, #111b3d 0%, transparent 45%),
        #050914 !important;
    color: #E5F3FF !important;
    max-width: 1500px !important;
    margin: auto !important;
}
h1, h2, h3 {
    color: #22D3EE !important;
}
.neon-panel {
    background: #0B1224;
    border: 1px solid #164E63;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 0 16px rgba(34, 211, 238, 0.07);
}
.muted {
    color: #94A3B8;
}
button.primary {
    background: linear-gradient(90deg, #0891B2, #7C3AED) !important;
    border: 0 !important;
}
footer {
    visibility: hidden;
}
"""


# ---------- ФИЛЬТРАЦИЯ ----------
def filter_data(search, category, source):
    if df.empty:
        return pd.DataFrame()

    result = df.copy()

    if search and search.strip():
        query = search.strip()
        searchable = (
            result["_title"].fillna("")
            + " "
            + result["_summary"].fillna("")
        )
        result = result[
            searchable.str.contains(
                re.escape(query), case=False, na=False
            )
        ]

    if category != "Все категории":
        result = result[result["_category"] == category]

    if source != "Все источники":
        result = result[result["_source"] == source]

    return result


def make_dashboard(search, category, source):
    empty_fig = px.bar(
        title="Нет данных для отображения"
    )
    empty_fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT
    )

    if df.empty:
        return (
            "### ⚠️ Датасет не найден",
            "Загрузите CSV рядом с app.py.",
            empty_fig,
            pd.DataFrame()
        )

    result = filter_data(search, category, source)

    total = len(df)
    visible = len(result)
    categories_count = df["_category"].nunique()
    sources_count = df["_source"].nunique()

    stats = f"""
<div class="neon-panel">

### ◈ DIGITAL HARASSMENT KZ

**{total:,}** — публикаций в датасете  
**{visible:,}** — найдено по текущим фильтрам  
**{categories_count:,}** — тематических категорий  
**{sources_count:,}** — источников

</div>
"""

    if result.empty:
        fig = empty_fig
    else:
        counts = (
            result["_category"]
            .value_counts()
            .rename_axis("Категория")
            .reset_index(name="Количество")
        )

        fig = px.bar(
            counts,
            x="Категория",
            y="Количество",
            color="Категория",
            color_discrete_sequence=[
                CYAN, PURPLE, "#0EA5E9",
                "#C026D3", "#6366F1", "#38BDF8"
            ],
            title="РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ"
        )

        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font_color=TEXT,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=100),
            xaxis_title="",
            yaxis_title="Количество публикаций"
        )
        fig.update_xaxes(tickangle=-25, gridcolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B")

    columns = {
        "Дата": "_date",
        "Заголовок": "_title",
        "Категория": "_category",
        "Источник": "_source",
        "Ссылка": "_url"
    }

    table = result[list(columns.values())].rename(
        columns={v: k for k, v in columns.items()}
    )

    return stats, f"### Найдено публикаций: {visible}", fig, table


# ---------- ИНТЕРФЕЙС ----------
if df.empty:
    categories = ["Все категории"]
    sources = ["Все источники"]
else:
    categories = ["Все категории"] + sorted(
        df["_category"].dropna().unique().tolist()
    )
    sources = ["Все источники"] + sorted(
        df["_source"].dropna().unique().tolist()
    )


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Digital Harassment KZ"
) as demo:

    gr.Markdown("""
# ◈ DIGITAL HARASSMENT KZ
### Аналитическая система мониторинга цифровых угроз
**Казахстан · Новостная аналитика · OSINT-подход к публичным публикациям**
""")

    with gr.Row():
        search = gr.Textbox(
            label="⌕ Поиск",
            placeholder="Введите слово или фразу..."
        )
        category = gr.Dropdown(
            choices=categories,
            value="Все категории",
            label="▦ Категория"
        )
        source = gr.Dropdown(
            choices=sources,
            value="Все источники",
            label="◉ Источник"
        )

    stats = gr.Markdown()
    status = gr.Markdown()

    with gr.Row():
        chart = gr.Plot(label="Аналитика по категориям")

    table = gr.Dataframe(
        label="База публикаций",
        interactive=False,
        wrap=True
    )

    gr.Markdown("""
---
**Примечание:** классификация по ключевым словам предварительная.
Количество публикаций отражает собранные материалы, а не фактическое
число случаев цифрового преследования.
""")

    inputs = [search, category, source]
    outputs = [stats, status, chart, table]

    for control in inputs:
        control.change(
            fn=make_dashboard,
            inputs=inputs,
            outputs=outputs
        )

    demo.load(
        fn=make_dashboard,
        inputs=inputs,
        outputs=outputs
    )


if __name__ == "__main__":
    demo.launch()
