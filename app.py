"""
Streamlit app for Reddit comment upvote prediction.
Provides a clean UI for users to input comments and select subreddits.
"""
import os
import streamlit as st
import pandas as pd
from transformers import pipeline
import torch


def load_subreddits_from_csv(csv_path: str = "subreddits.csv"):
    """
    Load unique subreddits from a CSV file.
    The CSV file should have a 'subreddit' column.
    Returns a list of unique subreddits sorted alphabetically.
    """
    # Try both project root and notebook directory
    paths_to_try = [
        csv_path,  # Current directory
        os.path.join("notebook", csv_path),  # Notebook directory
        os.path.join("..", csv_path),  # Parent directory
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "subreddit" in df.columns:
                    subreddits = df["subreddit"].unique().tolist()
                    return sorted(subreddits)
                else:
                    # If CSV exists but doesn't have 'subreddit' column, try first column
                    subreddits = df.iloc[:, 0].unique().tolist()
                    return sorted(subreddits)
            except Exception as e:
                st.warning(f"Error loading subreddits from {path}: {e}")
                continue
    
    return []


def filter_subreddits(search_term: str, subreddits: list):
    """Filter subreddits based on search term."""
    if not subreddits:
        return []
    if not search_term:
        return subreddits[:100]  # Show first 100 if no search
    search_lower = search_term.lower()
    filtered = [s for s in subreddits if search_lower in s.lower()]
    return filtered[:100]  # Limit to 100 results


def main():
    label2id = {'Controversial': 0, "Baseline": 1, "High Quality": 2, "Viral": 3}
    id2label = {v: k for k, v in label2id.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = pipeline('text-classification', model='roberta_classifier', device=device)
    classifier.model.config.label2id = label2id
    classifier.model.config.id2label = id2label
    st.set_page_config(
        page_title="Reddit Comment Upvote Predictor",
        layout="wide",
    )
    
    st.title("Reddit Comment Upvote Predictor")
    st.markdown("Enter a comment and select a subreddit to predict engagement.")
    
    # Load subreddits
    subreddits_list = load_subreddits_from_csv("subreddits.csv")
    
    if not subreddits_list:
        st.warning(
            "No subreddits CSV file found. Please create a `subreddits.csv` file "
            "with a 'subreddit' column."
        )
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Comment Input")
        comment = st.text_area(
            "Enter your Reddit comment",
            placeholder="Type your Reddit comment here...",
            height=200,
            help="Enter the comment text you want to analyze",
        )
    
    with col2:
        st.subheader("Subreddit Selection")
  
        selected_subreddit = st.selectbox(
            "Select Subreddit(s)",
            options=subreddits_list,
            placeholder="Choose subreddit...",
        )
     
    
    # Display preview and action button
    st.divider()
    
    if comment and selected_subreddit:
        st.subheader("Preview")
        
        st.markdown(f"**Selected Subreddit:** r/{selected_subreddit}")
        
        st.markdown(f"**Comment Length:** {len(comment)} characters")
        
        st.markdown("**Comment Preview:**")
        st.text_area(
            "Preview",
            value=comment,
            height=100,
            disabled=True,
            label_visibility="collapsed",
        )
        
        # Placeholder for prediction button
        st.markdown("---")
        if selected_subreddit:
            if st.button("Predict Engagement", type="primary", use_container_width=True):
                body = f"r/{selected_subreddit} {comment}"
                res = classifier([body])
                st.info(
                f"Your comment is {res[0]["label"]} with a confidence of {res[0]["score"] * 100:.2f}%"
                )
        else:
             st.info("Please fill in both the comment and select at least one subreddit to continue.")


if __name__ == "__main__":
    main()

