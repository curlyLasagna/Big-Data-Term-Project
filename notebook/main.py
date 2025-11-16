import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
    import matplotlib.pyplot as plt
    from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
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
            # "link_id",
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
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## EDA
    """)
    return


app._unparsable_cell(
    r"""
    def create_top_n_pie_chart(
        \"\"\"Returns a pie chart with the distribution of subreddits
        Allows us to analyze whether the dataset that we have is biased towards a single subreddit
        \"\"\"
    
        df: pd.DataFrame, subreddit_col=\"subreddit\", count_col=\"value\", top_n=10
    ):
        df = df.groupby(subreddit_col).size().reset_index(name=count_col)

        # 1. Sort and identify Top N
        df_sorted = df.sort_values(count_col, ascending=False).reset_index(
            drop=True
        )
        df_top_n = df_sorted.head(top_n).copy()

        # 2. Calculate the percentage
        total_sum = df_sorted[count_col].sum()
        df_top_n[\"percentage\"] = (df_top_n[count_col] / total_sum) * 100

        # 3. Calculate the 'Others' group value and percentage
        others_value = total_sum - df_top_n[count_col].sum()
        if others_value > 0:
            df_others = pd.DataFrame(
                [
                    {
                        subreddit_col: \"Others\",
                        count_col: others_value,
                        \"percentage\": (others_value / total_sum) * 100,
                    }
                ]
            )
            df_final = pd.concat([df_top_n, df_others], ignore_index=True)
        else:
            df_final = df_top_n.copy()

        # 4. Create the Altair Pie Chart with percentage
        base = (
            alt.Chart(df_final)
            .encode(
                theta=alt.Theta(\"percentage\", stack=True),
                color=alt.Color(field=subreddit_col, title=\"Subreddit\"),
                tooltip=[subreddit_col, count_col, \"percentage\"],
            )
            .properties(
                title=f\"Top {top_n} Subreddits by Comment Count (Percentage)\"
            )
        )

        pie = base.mark_arc(outerRadius=120).encode(
            order=alt.Order(\"percentage\", sort=\"descending\")
        )

        text = base.mark_text(radius=140).encode(
            text=alt.Text(\"percentage:Q\", format=\".1f\"),
            order=alt.Order(\"percentage\", sort=\"descending\"),
            color=alt.value(\"black\"),
        )

        return (pie + text).interactive()
    """,
    name="_"
)


@app.cell
def _(df):
    df["score"]
    return


@app.cell
def _(alt, pd):
    def create_score_histogram(score_series: pd.Series, title: str, low_score: float = None, high_score: float = None):
        # === PANDAS FILTERING STEP ===
        # 1. Start with the full series
        filtered_series = score_series.copy()
    
        # 2. Apply the filters directly to the series (more efficient)
        if low_score is not None:
            filtered_series = filtered_series[filtered_series >= low_score]
        
        if high_score is not None:
            filtered_series = filtered_series[filtered_series <= high_score]
        # =============================
    
        # 3. Convert the filtered Series to a DataFrame
        df = filtered_series.to_frame(name='score')

        # Create the Altair Density Plot (KDE)
        chart = alt.Chart(df).transform_density(
            density='score',
            # Define the names for the two new columns created by the transform
            as_=['score', 'density']
        ).mark_area(
            opacity=0.5,
            line=True
        ).encode(
            # X-axis is the original value, Y-axis is the calculated density
            x=alt.X('score:Q', title='Value'),
            y=alt.Y('density:Q', title='Probability Density'),
            tooltip=[
                alt.Tooltip('score:Q', title='Value'), 
                alt.Tooltip('density:Q', title='Density')
            ]
        ).properties(
            title=title
        ).interactive()

        return chart
    return (create_score_histogram,)


@app.cell
def _(create_score_histogram, df):
    create_score_histogram(df["score"], "scores", low_score=-200, high_score=-50)
    return


@app.cell
def _(alt, pd):
    def create_subreddit_histogram(df: pd.DataFrame):
        """Analyzes the same thing as the pie chart"""
        df = df.groupby('subreddit').size().reset_index(name='value').nlargest(10, "value")
        histogram = alt.Chart(df).mark_bar().encode(
            x=alt.X('subreddit', title='Subreddit', sort='-y'),
            y=alt.Y('value:Q', title='Number of Comments'),
            color=alt.Color('subreddit', title='Subreddit')
        ).properties(
            title='Comments Count per Subreddit'
        )

        return histogram
    return


@app.cell
def _(df):
    # Length of comments
    df["body"].map(lambda comment: len(comment)).describe()
    return


@app.cell
def _(df):
    df.groupby("subreddit").size().to_frame("value").describe()
    return


@app.cell
def _(create_top_n_pie_chart, df):
    create_top_n_pie_chart(df=df, top_n=10)
    return


@app.cell
def _(WordCloud, nltk, pd, wordcloud):
    def generate_stop_words(data: pd.Series, title: str):
        from nltk.corpus import stopwords
        import matplotlib.pyplot as plt
        nltk.download('stopwords')
        text = data.str.cat(sep = " ")
        word_cloud = WordCloud(
            max_words=1000,
            stopwords=stopwords.words('english'),
            collocations=False,
        ).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        plt.show()
    return


app._unparsable_cell(
    r"""
    # Generate word clouds for 
    low_score
    generate_stop_words(df[df[\"score\"] < ])
    """,
    name="_"
)


@app.cell
def _(df):
    df["score"].describe()
    return


@app.cell
def _(train_test):
    X_test, X_train, y_test, y_train = train_test
    return


if __name__ == "__main__":
    app.run()
