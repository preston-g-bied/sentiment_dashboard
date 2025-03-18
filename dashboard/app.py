"""
Sentiment Analysis Dashboard

A Streamlit dashboard for analyzing sentiment in social media posts.
"""

import os
import sys
import pandas as pd
import streamlit as st
import datetime
import glob
import plotly.graph_objects as go

# Add project root to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.sentiment_analyzer import SentimentAnalyzer
from src.visualization import (
    create_sentiment_time_series,
    create_sentiment_distribution_chart,
    create_top_topics_chart,
    generate_wordcloud
)

# Constants
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')


def load_data():
    """Load and combine all processed data files."""
    # Get all processed CSV files
    csv_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "processed_*.csv"))
    
    if not csv_files:
        st.error("No processed data files found. Please run data collection and processing scripts first.")
        st.stop()
    
    # Read and concatenate all files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert strings back to lists where needed
    combined_df['tokens'] = combined_df['tokens'].str.split()
    combined_df['hashtags'] = combined_df['hashtags'].str.split()
    
    # Convert timestamp to datetime
    combined_df['created_datetime'] = pd.to_datetime(combined_df['created_datetime'])
    
    # Run sentiment analysis if sentiment columns don't exist yet
    if 'sentiment_label' not in combined_df.columns:
        st.info("Running sentiment analysis on the data...")
        analyzer = SentimentAnalyzer()
        combined_df = analyzer.analyze_dataframe(combined_df)
    
    return combined_df


def filter_data(df, start_date, end_date, selected_sentiment=None, search_term=None):
    """Filter data based on user selections."""
    # Filter by date range
    filtered_df = df[(df['created_datetime'] >= start_date) & 
                      (df['created_datetime'] <= end_date)]
    
    # Filter by sentiment if selected and if sentiment_label column exists
    if selected_sentiment and selected_sentiment != "All" and 'sentiment_label' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment.lower()]
    
    # Filter by search term if provided
    if search_term:
        # Search in title, body, and tokens
        term_lower = search_term.lower()
        title_mask = filtered_df['title'].str.lower().str.contains(term_lower, na=False)
        body_mask = filtered_df['body'].fillna('').str.lower().str.contains(term_lower, na=False)
        
        # Check if term is in tokens list
        token_mask = filtered_df['tokens'].apply(
            lambda tokens: any(term_lower in token.lower() for token in tokens if token)
        )
        
        # Combine masks
        filtered_df = filtered_df[title_mask | body_mask | token_mask]
    
    return filtered_df


def run_dashboard():
    """Main function to run the Streamlit dashboard."""
    # Set page config
    st.set_page_config(
        page_title="Sentiment Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Dashboard title
    st.title("Social Media Sentiment Analysis Dashboard")
    st.write("Analyze sentiment trends in social media posts.")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range selector
    min_date = df['created_datetime'].min().date()
    max_date = df['created_datetime'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Convert to datetime for filtering
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Sentiment filter
    sentiment_options = ["All", "Positive", "Neutral", "Negative"]
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)
    
    # Search term
    search_term = st.sidebar.text_input("Search Term")
    
    # Apply filters
    filtered_df = filter_data(
        df, 
        start_datetime, 
        end_datetime, 
        selected_sentiment,
        search_term
    )
    
    # Display filters applied
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Posts Found:** {len(filtered_df)}")
    
    # If no data matches filters
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
        return
    
    # Dashboard Layout
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment Overview", 
        "Time Analysis", 
        "Topic Analysis",
        "Word Cloud"
    ])
    
    # Tab 1: Sentiment Overview
    with tab1:
        st.header("Sentiment Overview")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            fig_dist = create_sentiment_distribution_chart(filtered_df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.subheader("Summary Statistics")
            
            # Calculate metrics
            total_posts = len(filtered_df)
            avg_sentiment = filtered_df['sentiment_compound'].mean()
            pos_percentage = (filtered_df['sentiment_label'] == 'positive').mean() * 100
            neg_percentage = (filtered_df['sentiment_label'] == 'negative').mean() * 100
            neu_percentage = (filtered_df['sentiment_label'] == 'neutral').mean() * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            col2.metric("Positive Posts", f"{pos_percentage:.1f}%")
            col3.metric("Negative Posts", f"{neg_percentage:.1f}%")
            
            # Display most positive and negative posts
            st.subheader("Most Extreme Sentiments")
            
            # Most positive post
            most_positive = filtered_df.loc[filtered_df['sentiment_compound'].idxmax()]
            st.write("**Most Positive Post:**")
            st.info(
                f"**Score:** {most_positive['sentiment_compound']:.2f}\n\n"
                f"**Title:** {most_positive['title']}"
            )
            
            # Most negative post
            most_negative = filtered_df.loc[filtered_df['sentiment_compound'].idxmin()]
            st.write("**Most Negative Post:**")
            st.error(
                f"**Score:** {most_negative['sentiment_compound']:.2f}\n\n"
                f"**Title:** {most_negative['title']}"
            )
    
    # Tab 2: Time Analysis
    with tab2:
        st.header("Sentiment Over Time")
        
        # Time unit selector
        time_unit = st.selectbox(
            "Time Granularity",
            options=["Hour", "Day", "Week", "Month"],
            index=1  # Default to Day
        )
        
        # Map selection to pandas time frequency
        time_map = {
            "Hour": "H",
            "Day": "D",
            "Week": "W",
            "Month": "M"
        }
        
        # Aggregate sentiment by selected time period
        time_aggregated = analyzer.aggregate_sentiment_by_time(
            filtered_df, 
            time_unit=time_map[time_unit]
        )
        
        # Create time series chart
        if not time_aggregated.empty:
            fig_time = create_sentiment_time_series(time_aggregated)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Show the aggregated data
            st.subheader("Aggregated Data")
            st.dataframe(
                time_aggregated.rename(columns={
                    'time_period': 'Time Period',
                    'sentiment_compound': 'Avg. Sentiment',
                    'post_count': 'Post Count'
                }).sort_values('Time Period', ascending=False)
            )
        else:
            st.warning("Not enough data to create time series visualization.")
    
    # Tab 3: Topic Analysis
    with tab3:
        st.header("Topic Analysis")
        
        # Number of topics selector
        n_topics = st.slider("Number of Topics", min_value=5, max_value=20, value=10)
        
        # Identify trending topics
        top_topics = analyzer.identify_trending_topics(filtered_df, n_topics=n_topics)
        
        if not top_topics.empty:
            # Create topics chart
            fig_topics = create_top_topics_chart(top_topics)
            st.plotly_chart(fig_topics, use_container_width=True)
            
            # Show the topics data
            st.subheader("Topic Data")
            st.dataframe(
                top_topics.rename(columns={
                    'topic': 'Topic',
                    'mention_count': 'Mentions',
                    'avg_sentiment': 'Avg. Sentiment',
                    'sentiment_label': 'Sentiment'
                }).sort_values('Mentions', ascending=False)
            )
        else:
            st.warning("Not enough data to identify topics.")
    
    # Tab 4: Word Cloud
    with tab4:
        st.header("Word Cloud Visualization")
        
        # Word cloud sentiment filter
        wc_sentiment = st.radio(
            "Show words for sentiment:",
            options=["All", "Positive", "Neutral", "Negative"],
            horizontal=True
        )
        
        # Generate word cloud
        sentiment_filter = None if wc_sentiment == "All" else wc_sentiment.lower()
        wordcloud_img = generate_wordcloud(filtered_df, sentiment_filter=sentiment_filter)
        
        if wordcloud_img:
            st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)
        else:
            st.warning("Not enough text data to generate word cloud.")


if __name__ == "__main__":
    run_dashboard()