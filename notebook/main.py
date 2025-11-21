import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer
    from transformers import RobertaTokenizerFast, TrainingArguments, Trainer
    import matplotlib.pyplot as plt
    from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
    import pandas as pd
    import altair as alt
    from wordcloud import WordCloud
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    import numpy as np
    alt.data_transformers.enable("vegafusion")
    return WordCloud, alt, load_dataset, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


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
    ).filter(lambda x: x["score"] != 1)


    df = dataset.take(100000)
    df = pd.DataFrame(list(df))
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## EDA
    """)
    return


@app.cell(hide_code=True)
def _(alt, pd):
    def create_top_n_pie_chart(
        df: pd.DataFrame, subreddit_col="subreddit", count_col="value", top_n=10
    ) -> alt.Chart:
        """
        Returns a pie chart with the distribution of subreddits
        Allows us to analyze whether the dataset that we have is biased towards a single subreddit
        """
        df = df.groupby(subreddit_col).size().reset_index(name=count_col)

        # 1. Sort and identify Top N
        df_sorted = df.sort_values(count_col, ascending=False).reset_index(
            drop=True
        )
        df_top_n = df_sorted.head(top_n).copy()

        # 2. Calculate the percentage
        total_sum = df_sorted[count_col].sum()
        df_top_n["percentage"] = (df_top_n[count_col] / total_sum) * 100

        # 3. Calculate the 'Others' group value and percentage
        others_value = total_sum - df_top_n[count_col].sum()
        if others_value > 0:
            df_others = pd.DataFrame(
                [
                    {
                        subreddit_col: "Others",
                        count_col: others_value,
                        "percentage": (others_value / total_sum) * 100,
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
                theta=alt.Theta("percentage", stack=True),
                color=alt.Color(field=subreddit_col, title="Subreddit"),
                tooltip=[subreddit_col, count_col, "percentage"],
            )
            .properties(
                title=f"Top {top_n} Subreddits by Comment Count (Percentage)"
            )
        )

        pie = base.mark_arc(outerRadius=120).encode(
            order=alt.Order("percentage", sort="descending")
        )

        text = base.mark_text(radius=140).encode(
            text=alt.Text("percentage:Q", format=".1f"),
            order=alt.Order("percentage", sort="descending"),
            color=alt.value("black"),
        )

        return (pie + text).interactive()
    return (create_top_n_pie_chart,)


@app.cell(hide_code=True)
def _(alt, pd):
    def create_word_count_histogram(df: pd.DataFrame, column: str, title: str):
        """Visualizes a histogram of word counts for the specified column in the DataFrame."""
        # Count words in the specified column
        word_counts = df[column].apply(lambda x: len(x.split()))

        # Create a DataFrame from the word counts
        count_df = pd.DataFrame({"word_count": word_counts})

        # Create the Altair histogram
        chart = alt.Chart(count_df).mark_bar().encode(
            x=alt.X('word_count', bin=alt.Bin(maxbins=500), title='Word Count'),
            y=alt.Y('count()', title='Frequency'),
            tooltip=[alt.Tooltip('word_count', bin=True), 'count()']
        ).properties(
            title=title
        ).interactive()

        return chart
    return (create_word_count_histogram,)


@app.cell(hide_code=True)
def _(pd):
    def count_words_with_body(df: pd.DataFrame) -> pd.DataFrame:
        """Count words in the 'body' column and return a DataFrame with the body and corresponding word counts."""
        word_counts = df['body'].apply(lambda x: len(x.split()))
        return pd.DataFrame({"body": df['body'], "word_count": word_counts})
    return (count_words_with_body,)


@app.cell(hide_code=True)
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
        chart = alt.Chart(df).mark_bar().encode(
        # Bin the 'Value' column to create the histogram bins.
        # 'maxbins=30' specifies the maximum number of bars/bins.
        x=alt.X('score', bin=alt.Bin(maxbins=200), title='Value Range'),

        # Use 'count()' to get the frequency (bar height) for each bin.
        y=alt.Y('count()', title='Frequency'),

        # Add tooltips for interaction
        tooltip=[
            alt.Tooltip('score', bin=True, title='Value Range'), 
            'count()'
        ]
        ).properties(
            title=title
        ).interactive()

        return chart
    return (create_score_histogram,)


@app.cell(hide_code=True)
def _(pd):
    def get_top_frequent_words(df: pd.DataFrame, top_n: int = 20):
        from collections import Counter
        import re

        # Combine all comments into a single string
        text = " ".join(df["body"].astype(str).tolist())

        # Use regex to remove punctuation and make everything lowercase
        words = re.findall(r'\b\w+\b', text.lower())

        # Count word frequencies
        word_counts = Counter(words)

        # Get the top N most common words
        top_words = word_counts.most_common(top_n)

        return pd.DataFrame(top_words, columns=["word", "frequency"])
    return


@app.cell(hide_code=True)
def _(WordCloud, pd):
    def generate_stop_words(data: pd.Series, title: str):
        import nltk
        from nltk.corpus import stopwords
        import matplotlib.pyplot as plt
        import re
        nltk.download('stopwords')
        text = data.str.cat(sep = " ")
        # Remove HTML tags and filter words with less than 5 characters
        cleaned_text = re.sub(r'<.*?>', '', text)  # Remove HTML content
        words = re.findall(r'\b\w{5,}\b', cleaned_text)  # Match words with 5 or more characters

        word_cloud = WordCloud(
            max_words=1000,
            stopwords=stopwords.words('english').append("like"),
            collocations=False,
            scale=2
        ).generate(" ".join(words))  # Generate word cloud from filtered words

        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        plt.savefig(title, format='png', bbox_inches='tight')
        plt.show()
    return (generate_stop_words,)


@app.cell
def _(alt, pd):
    def create_subreddit_histogram(df: pd.DataFrame):
        """Histogram of the subreddits"""
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
def _(alt, pd):
    def create_subreddit_avg_votes_histogram(data: pd.DataFrame):
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('subreddit', axis=alt.Axis(labels=False)),
            y=alt.Y('score:Q', title="Average score"),
            color=alt.Color('subreddit', title='Subreddit', legend=None)
        ).properties(
            title="Subreddit average upvotes"
        )

        return chart
    return (create_subreddit_avg_votes_histogram,)


@app.cell(hide_code=True)
def _(alt, count_words_with_body, pd):
    def create_word_counts_chart(df: pd.DataFrame, title: str) -> alt.Chart:
        """Visualizes a bar chart of word counts and their corresponding bodies."""
        # Call the count_words_with_body function to get the word counts
        word_counts_df = count_words_with_body(df)

        # Create an Altair bar chart
        chart = alt.Chart(word_counts_df).mark_bar().encode(
            x=alt.X('word_count', title='Word Count'),
            y=alt.Y('count()', title='Frequency', scale=alt.Scale(domain=[0, 55])),
            tooltip=['body', 'word_count']
        ).properties(
            title=title
        ).interactive()

        return chart
    return (create_word_counts_chart,)


@app.cell
def _(df):
    df.groupby(["subreddit"])["score"].mean().reset_index()
    return


@app.cell
def _(df):
    breh = df.groupby(["subreddit"])["score"].mean().reset_index()
    return (breh,)


@app.cell
def _(breh, create_subreddit_avg_votes_histogram):
    create_subreddit_avg_votes_histogram(breh).save(fp="subreddit_avg.png", scale_factor=2)
    return


@app.cell
def _(create_word_counts_chart, df):
    create_word_counts_chart(df, "Comment length distribution").save(fp="comment_length_distribution.png", scale_factor=2)
    return


@app.cell
def _(df):
    df["score"].describe()
    return


@app.cell
def _(create_word_count_histogram, df):
    create_word_count_histogram(df=df, column="body", title="huh")
    return


@app.cell
def _(count_words_with_body, df):
    count_words_with_body(df).describe()
    return


@app.cell
def _(create_score_histogram, df):
    create_score_histogram(df["score"], "scores", low_score=-30, high_score=100)
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
    create_top_n_pie_chart(df=df, top_n=10).save(fp="subreddit_pie.png", scale_factor=2)
    return


@app.cell
def _():
    return


@app.cell
def _(df, generate_stop_words):
    # Generate word clouds for comments with negative votes and positive votes
    negative_scores_df = df[df["score"] > 5]
    positive_scores_df = df[df["score"] < -3]
    generate_stop_words(negative_scores_df["body"], title="Words commonly found with negatively voted comments")
    return (positive_scores_df,)


@app.cell
def _(generate_stop_words, positive_scores_df):
    generate_stop_words(positive_scores_df["body"], title="Words commonly found with positively voted comments")
    return


@app.cell
def _(df):
    from denseweight import DenseWeight
    dw = DenseWeight(alpha=1.0)
    dw.fit(df["score"].to_numpy())
    dw([2])
    return


@app.cell
def _(df, np, pd):
    #define bins and labels
    bins = [-np.inf, -1.0, 0.5, 1.5, np.inf]
    labels = ["Controversial", "Baseline", "High Quality", "Viral"]

    df_binned = df.copy()

    #grouped by subreddit
    groups = df_binned.groupby("subreddit")["score"]
    mean = groups.transform("mean")
    std = groups.transform("std").replace(0,np.nan)
    raw_z = (df_binned["score"] - mean) / std

    #if std == 0, treat z as 0
    df_binned["z_score"] = raw_z.fillna(0.0)
    #bin
    df_binned["quality_label"] = pd.cut(df_binned["z_score"], bins=bins, labels=labels)
    df_binned

    return (df_binned,)


@app.cell
def _(df_binned):
    ordinal_map = {
        "Controversial": 0,
        "Baseline": 1,
        "High Quality": 2,
        "Viral": 3,
    }
    df_binned["quality_num"] = df_binned["quality_label"].map(ordinal_map)

    return


@app.cell
def _(df_binned):
    df_binned["quality_label"].value_counts()
    return


@app.cell
def _(df_binned, plt):
    plt.figure(figsize=(10, 6))
    df_binned["quality_label"].value_counts().sort_index().plot(kind='bar', color=['crimson', 'steelblue', 'mediumseagreen', 'gold'])
    plt.title('Distribution of Scores Binned', fontsize=14, fontweight='bold')
    plt.xlabel('Quality Label', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
