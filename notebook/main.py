import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
    from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    import altair as alt
    from wordcloud import WordCloud
    return WordCloud, alt, load_dataset, mo, pd


@app.cell
def _(load_dataset, pd):
    dataset = load_dataset(
        "fddemarco/pushshift-reddit-comments", split="train", streaming=True
    ).remove_columns(
        column_names=[
            "link_id",
            "subreddit_id",
            "id",
            "created_utc",
            "controversiality",
        ]
    )
    df = dataset.take(100000)
    df = pd.DataFrame(list(df))
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## EDA""")
    return


@app.cell
def _(alt, pd):
    def create_top_n_pie_chart(df: pd.DataFrame, subreddit_col='subreddit', count_col='value', top_n=10):
        # 1. Sort and identify Top N
        df_sorted = df.sort_values(count_col, ascending=False).reset_index(drop=True)
        df_top_n = df_sorted.head(top_n).copy()

        # 2. Calculate the 'Others' group value
        total_sum = df_sorted[count_col].sum()
        top_n_sum = df_top_n[count_col].sum()
        others_value = total_sum - top_n_sum

        # 3. Create the 'Others' row and combine the data
        if others_value > 0:
            df_others = pd.DataFrame([{subreddit_col: 'Others', count_col: others_value}])
            df_final = pd.concat([df_top_n, df_others], ignore_index=True)
        else:
            # If there are 10 or fewer, no 'Others' group is needed
            df_final = df_top_n.copy()

        # 4. Create the Altair Pie Chart
        base = alt.Chart(df_final).encode(
            # Encode angle using the count column
            theta=alt.Theta(count_col, stack=True),
            # Add tooltips for hover interaction
            tooltip=[subreddit_col, count_col]
        ).properties(
            title=f"Top {top_n} Subreddits by Comment Count"
        )

        # Pie/Donut Chart
        pie = base.mark_arc(outerRadius=120).encode(
            # Encode color using the subreddit column
            color=alt.Color(subreddit_col, title="Subreddit"),
            # Set sort order to ensure segments are consistent (largest to smallest)
            order=alt.Order(count_col, sort="descending")
        )

        # Optional: Text labels showing the count
        text = base.mark_text(radius=140).encode(
            text=alt.Text(count_col),
            order=alt.Order(count_col, sort="descending"),
            color=alt.value("black")
        )

        return (pie).interactive()
    return (create_top_n_pie_chart,)


@app.cell
def _(create_top_n_pie_chart, df):
    create_top_n_pie_chart(df=df.groupby("subreddit").size().to_frame('value'))
    return


@app.cell
def _(WordCloud):
    WordCloud(max_words=1000).generate()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
