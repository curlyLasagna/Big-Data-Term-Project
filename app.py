"""
Streamlit app for Reddit comment upvote prediction.
Provides a clean UI for users to input comments and select subreddits.
"""
import os
import streamlit as st
import pandas as pd


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
        
        # Search input
        search_term = st.text_input(
            "Search Subreddit",
            placeholder="Type to search...",
            help="Type to filter the list of available subreddits",
        )
        
        # Filter subreddits based on search
        filtered_subreddits = filter_subreddits(search_term, subreddits_list)
        
        # Multiselect for subreddit selection
        if filtered_subreddits:
            selected_subreddits = st.multiselect(
                "Select Subreddit(s)",
                options=filtered_subreddits,
                default=None,
                help="Choose one or more subreddits from the filtered list. You can select multiple to compare predictions.",
                placeholder="Choose subreddit(s)...",
            )
        else:
            selected_subreddits = []
            if subreddits_list:
                st.info("No subreddits match your search. Try a different term.")
            else:
                st.info("No subreddits available.")
    
    # Display preview and action button
    st.divider()
    
    if comment and selected_subreddits:
        st.subheader("Preview")
        col_preview1, col_preview2 = st.columns(2)
        
        with col_preview1:
            if len(selected_subreddits) == 1:
                st.markdown(f"**Selected Subreddit:** r/{selected_subreddits[0]}")
            else:
                subreddit_list = ", ".join([f"r/{s}" for s in selected_subreddits])
                st.markdown(f"**Selected Subreddits:** {subreddit_list}")
        
        with col_preview2:
            st.markdown(f"**Comment Length:** {len(comment)} characters")
            st.markdown(f"**Subreddits Selected:** {len(selected_subreddits)}")
        
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
        if st.button("Predict Engagement", type="primary", use_container_width=True):
            if len(selected_subreddits) == 1:
                st.info(
                    f"Prediction functionality will be added once the model is trained. "
                    f"The model will classify the comment for r/{selected_subreddits[0]} as: "
                    "Controversial, Baseline, High Quality, or Viral."
                )
            else:
                st.info(
                    f"Prediction functionality will be added once the model is trained. "
                    f"The model will classify the comment for {len(selected_subreddits)} subreddit(s) as: "
                    "Controversial, Baseline, High Quality, or Viral."
                )
    elif comment or selected_subreddits:
        st.info("Please fill in both the comment and select at least one subreddit to continue.")
    else:
        st.info("Enter a comment and select at least one subreddit to get started.")


if __name__ == "__main__":
    main()

