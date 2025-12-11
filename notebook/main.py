import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        pipeline
    )
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib.pyplot as plt
    from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
    from datasets import Dataset
    import pandas as pd
    import altair as alt
    from wordcloud import WordCloud
    import torch
    import numpy as np
    import pandas as pd

    alt.data_transformers.enable("vegafusion")
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Dataset,
        StandardScaler,
        Trainer,
        TrainingArguments,
        WordCloud,
        alt,
        compute_class_weight,
        load_dataset,
        mo,
        np,
        pd,
        pipeline,
        torch,
        train_test_split,
    )


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
    )


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
        chart = (
            alt.Chart(count_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "word_count", bin=alt.Bin(maxbins=500), title="Word Count"
                ),
                y=alt.Y("count()", title="Frequency"),
                tooltip=[alt.Tooltip("word_count", bin=True), "count()"],
            )
            .properties(title=title)
            .interactive()
        )

        return chart
    return (create_word_count_histogram,)


@app.cell(hide_code=True)
def _(pd):
    def count_words_with_body(df: pd.DataFrame) -> pd.DataFrame:
        """Count words in the 'body' column and return a DataFrame with the body and corresponding word counts."""
        word_counts = df["body"].apply(lambda x: len(x.split()))
        return pd.DataFrame({"body": df["body"], "word_count": word_counts})
    return (count_words_with_body,)


@app.cell(hide_code=True)
def _(alt, pd):
    def create_score_histogram(
        score_series: pd.Series,
        title: str,
        low_score: float = None,
        high_score: float = None,
    ):
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
        df = filtered_series.to_frame(name="score")

        # Create the Altair Density Plot (KDE)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                # Bin the 'Value' column to create the histogram bins.
                # 'maxbins=30' specifies the maximum number of bars/bins.
                x=alt.X("score", bin=alt.Bin(maxbins=100), title="Value Range"),
                # Use 'count()' to get the frequency (bar height) for each bin.
                y=alt.Y("count()", title="Frequency"),
                # Add tooltips for interaction
                tooltip=[
                    alt.Tooltip(
                        "score", bin=alt.Bin(maxbins=100), title="Value Range"
                    ),
                    "count()",
                ],
            )
            .properties(title=title)
            .interactive()
        )

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
        words = re.findall(r"\b\w+\b", text.lower())

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

        nltk.download("stopwords")
        text = data.str.cat(sep=" ")
        # Remove HTML tags and filter words with less than 5 characters
        cleaned_text = re.sub(r"<.*?>", "", text)  # Remove HTML content
        words = re.findall(
            r"\b\w{5,}\b", cleaned_text
        )  # Match words with 5 or more characters

        word_cloud = WordCloud(
            max_words=1000,
            stopwords=stopwords.words("english").append("like"),
            collocations=False,
            scale=2,
        ).generate(" ".join(words))  # Generate word cloud from filtered words

        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.savefig(title, format="png", bbox_inches="tight")
        plt.show()
    return (generate_stop_words,)


@app.cell(hide_code=True)
def _(alt, pd):
    def create_subreddit_histogram(df: pd.DataFrame):
        """Histogram of the subreddits"""
        df = (
            df.groupby("subreddit")
            .size()
            .reset_index(name="value")
            .nlargest(10, "value")
        )
        histogram = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("subreddit", title="Subreddit", sort="-y"),
                y=alt.Y("value:Q", title="Number of Comments"),
                color=alt.Color("subreddit", title="Subreddit"),
            )
            .properties(title="Comments Count per Subreddit")
        )

        return histogram
    return


@app.cell
def _(alt, df, pd):
    def create_subreddit_avg_votes_histogram(data: pd.DataFrame):
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("subreddit", axis=alt.Axis(labels=False)),
                y=alt.Y("score:Q", title="Average score"),
                color=alt.Color("subreddit", title="Subreddit", legend=None),
            )
            .properties(title="Subreddit average upvotes")
        )

        return chart

    create_subreddit_avg_votes_histogram(df).save(fp="avg_votes.png", scale_factor=2)
    return (create_subreddit_avg_votes_histogram,)


@app.cell(hide_code=True)
def _(alt, count_words_with_body, pd):
    def create_word_counts_chart(df: pd.DataFrame, title: str) -> alt.Chart:
        """Visualizes a bar chart of word counts and their corresponding bodies."""
        # Call the count_words_with_body function to get the word counts
        word_counts_df = count_words_with_body(df)

        # Create an Altair bar chart
        chart = (
            alt.Chart(word_counts_df)
            .mark_bar()
            .encode(
                x=alt.X("word_count", title="Word Count"),
                y=alt.Y(
                    "count()", title="Frequency", scale=alt.Scale(domain=[0, 55])
                ),
                tooltip=["body", "word_count"],
            )
            .properties(title=title)
            .interactive()
        )

        return chart
    return (create_word_counts_chart,)


@app.cell
def _(df):
    df.groupby(["subreddit"])["score"].mean().reset_index().describe()
    return


@app.cell
def _(df):
    breh = df.groupby(["subreddit"])["score"].mean().reset_index()
    return (breh,)


@app.cell
def _(breh, create_subreddit_avg_votes_histogram):
    create_subreddit_avg_votes_histogram(breh).save(
        fp="subreddit_avg.png", scale_factor=2
    )
    return


@app.cell
def _(create_word_counts_chart, df):
    create_word_counts_chart(df, "Comment length distribution").save(
        fp="comment_length_distribution.png", scale_factor=2
    )
    return


@app.cell
def _(df):
    df["score"].describe()
    return


@app.cell
def _(create_word_count_histogram, df):
    create_word_count_histogram(
        df=df[df["body"].str.split().str.len() < 300],
        column="body",
        title="Comment Word Count Distribution",
    )
    # .save(fp="word_count_distribution.png", scale_factor=2)
    return


@app.cell
def _(df):
    df[df["body"].str.split().str.len() < 200]
    return


@app.cell
def _(count_words_with_body, df):
    count_words_with_body(df)
    return


@app.cell
def _(create_score_histogram, df):
    create_score_histogram(
        df["score"], "scores", low_score=-10, high_score=50
    ).save(fp="score_frequency.png", scale_factor=2)
    return


@app.cell
def _(df):
    df.groupby("subreddit").size().to_frame("value").describe()
    return


@app.cell
def _(create_top_n_pie_chart, df):
    create_top_n_pie_chart(df=df, top_n=10).save(
        fp="subreddit_pie.png", scale_factor=2
    )
    return


@app.cell
def _(df, generate_stop_words):
    # Generate word clouds for comments with negative votes and positive votes
    negative_scores_df = df[df["score"] > 5]
    positive_scores_df = df[df["score"] < -3]
    generate_stop_words(
        negative_scores_df["body"],
        title="Words commonly found with negatively voted comments",
    )
    return (positive_scores_df,)


@app.cell
def _(generate_stop_words, positive_scores_df):
    generate_stop_words(
        positive_scores_df["body"],
        title="Words commonly found with positively voted comments",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preprocessing
    """)
    return


@app.function
def clean_comments(text: str):
    import re

    # --- Collapse spaced-out URL patterns specifically ---
    # e.g., "h t t p : / /" -> "http://"
    text = re.sub(
        r"h\s*t\s*t\s*p\s*:\s*/\s*/", "http://", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"h\s*t\s*t\s*p\s*s\s*:\s*/\s*/", "https://", text, flags=re.IGNORECASE
    )

    # Collapse domain patterns: "example . com" -> "example.com"
    text = re.sub(r"(\w)\s*\.\s*(\w)", r"\1.\2", text)

    # --- Collapse spaced HTML entities ONLY ---
    # e.g., "& g t ;" -> "&gt;"
    text = re.sub(r"&\s*([a-zA-Z0-9]+)\s*;", r"&\1;", text)

    # --- Remove HTML entities ---
    text = re.sub(r"&[#0-9a-zA-Z]+;", "", text)

    # --- Remove markdown links ---
    text = re.sub(r"\[\s*[^]]+?\s*\]\s*\(\s*[^)]+?\s*\)", "", text)

    # --- Collapse any now-normalized URL after fixing spacing ---
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(
        r"\b[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?", "", text, flags=re.IGNORECASE
    )

    # --- Clean up leftover whitespace ---
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.cell
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tokenizing the input
    """)
    return


@app.cell
def _(tokenizer):
    tokenizer.save_pretrained("roberta_classifier")
    return


@app.cell
def _(AutoTokenizer, StandardScaler, np, pd):
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def preprocess_function(df: pd.DataFrame):
        def filter_word_count(df: pd.DataFrame):
            # Filter out rows with body word count < 300
            clean_df = df[df["body"].str.split().str.len() < 300]
            clean_df["body"] = clean_df["body"].apply(clean_comments)
            return clean_df

        def filter_junk_comments(df: pd.DataFrame):
            JUNK_STRINGS = [
                ",",
                ".",
                "http://",
                "http:// /",
                "no .",
                r"# 3232 ; \ _ # 3232 ;",
                "yes .",
                "no",
            ]
            not_junk_mask = ~df["body"].isin(JUNK_STRINGS)
            is_not_empty_or_whitespace = df["body"].str.strip() != ""
            combined_mask = not_junk_mask & is_not_empty_or_whitespace

            return df[combined_mask].copy()

        def create_z_score_labels(df: pd.DataFrame):
            def group_scaler(x):
                # Handle groups with only 1 sample or empty groups to prevent errors
                if len(x) < 2:
                    return np.zeros(len(x))

                scaler = StandardScaler()
                # reshape(-1, 1) is required because StandardScaler expects 2D array
                return scaler.fit_transform(x.values.reshape(-1, 1)).flatten()

            # Apply the scaler per subreddit
            df["z_score"] = df.groupby("subreddit")["score"].transform(
                group_scaler
            )

            # --- 2. Create Class Labels (Binning) ---
            def get_label(z):
                # Controversial
                if z < -1.5:
                    return 0
                # Baseline (Majority)
                elif z <= 1.0:
                    return 1
                # High Quality
                elif z <= 3.0:
                    return 2
                # Viral
                else:
                    return 3

            df["labels"] = df["z_score"].apply(get_label)

            # Drop the temporary z_score column (and score if not needed for features)
            return df.drop(columns=["z_score"])

        def text_feature_extraction(df: pd.DataFrame):
            df["text"] = (
                "r/" + df["subreddit"].astype(str) + " " + df["body"].astype(str)
            )
            return df

        def tokenize_text(df: pd.DataFrame, column: str):

            encoded_tokens = tokenizer(
                list(df[column]),
                truncation=True,
                padding="max_length",
                max_length=325,
            )
            df["input_ids"] = encoded_tokens["input_ids"]
            df["attention_mask"] = encoded_tokens["attention_mask"]

            return df

        return tokenize_text(
            text_feature_extraction(
                create_z_score_labels(filter_junk_comments(filter_word_count(df)))
            ),
            column="text",
        )[["text", "input_ids", "attention_mask", "labels"]]
    return preprocess_function, tokenizer


@app.cell
def _(df, preprocess_function):
    processed_df = preprocess_function(df)
    return (processed_df,)


@app.cell
def _(processed_df, train_test_split):
    train_df, test_df = train_test_split(
        processed_df, test_size=0.3, stratify=processed_df["labels"], random_state=42
    )
    return test_df, train_df


@app.cell
def _(train_df):
    train_df
    return


@app.cell
def _(AutoModelForSequenceClassification, TrainingArguments, torch):
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        # Directory to save model checkpoints
        output_dir="./results",  
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=3,
        # Training batch size
        per_device_train_batch_size=32,
        # Evaluation batch size
        per_device_eval_batch_size=32,
        learning_rate=2e-5,  # Standard learning rate for fine-tuning BERT
        weight_decay=0.01,  # Regularization to prevent overfitting
        save_total_limit=2,  # Save checkpoints every 'n' steps
        load_best_model_at_end=True,  # Load the best model based on evaluation metrics
        fp16=True,  # Enable mixed precision for faster training
        logging_steps=100,  # Log metrics every 'n' steps
    )
    return device, model, training_args


@app.cell
def _(Dataset, Trainer, model, test_df, train_df, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_pandas(train_df),
        eval_dataset=Dataset.from_pandas(test_df),
    )

    trainer.train()
    return


@app.cell
def _(
    CustomLossTrainer,
    Dataset,
    compute_class_weight,
    compute_metrics,
    device,
    model,
    np,
    test_df,
    torch,
    train_df,
    training_args,
):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["labels"]),
        y=train_df["labels"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    custom_trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_pandas(train_df),
        eval_dataset=Dataset.from_pandas(test_df),
        loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device),
        compute_metrics=compute_metrics
    )
    return (custom_trainer,)


@app.cell
def _(custom_trainer):
    custom_trainer.train()
    custom_trainer.save_model('roberta_classifier')
    return


@app.cell
def _(custom_trainer):
    custom_trainer.evaluate()
    return


@app.cell
def _(Trainer):
    class CustomLossTrainer(Trainer):
        def __init__(self, *args, loss_fn=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fn = loss_fn

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")

            # Forward pass (now faster, as it skips internal loss calc)
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Compute your custom weighted loss
            loss = self.loss_fn(logits, labels)

            # If using newer transformers with gradient accumulation, you might need to handle scaling:
            if num_items_in_batch is not None:
                return loss / num_items_in_batch 

            return (loss, outputs) if return_outputs else loss
    return (CustomLossTrainer,)


@app.cell
def _(device, pipeline):
    label2id = {'Controversial': 0, "Baseline": 1, "High Quality": 2, "Viral": 3}
    id2label = {v: k for k, v in label2id.items()}

    classifier = pipeline('text-classification', model='roberta_classifier', device=device)
    classifier.model.config.label2id = label2id
    classifier.model.config.id2label = id2label

    classifier(["r/starcraft gg to you too ;)"])
    return


@app.cell
def _(df):
    # Extract unique subreddits and save to CSV for Streamlit app
    unique_subreddits = df["subreddit"].unique()
    subreddit_df = pd.DataFrame({"subreddit": sorted(unique_subreddits)})
    subreddit_df.to_csv("../subreddits.csv", index=False)
    print(f"✓ Saved {len(unique_subreddits)} unique subreddits to subreddits.csv")
    return


@app.cell
def _():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    
        # Calculate accuracy
        acc = accuracy_score(labels, preds)
    
        # Calculate precision, recall, and F1 (weighted accounts for class imbalance)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    return (compute_metrics,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
