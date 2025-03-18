"""
Sentiment Analyzer Module

This module provides functions for analyzing sentiment in text data using VADER sentiment analysis.
"""

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# download VADER lexicon if not already downloaded
nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """Class for performing sentiment analysis on text data."""

    def __init__(self):
        """Initialize the sentiment analyzer with VADER."""
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """
        Analyze the sentiment of a text string.

        Args:
            text: String text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0,
                'pos': 0,
                'neu': 0,
                'neg': 0
            }
        
        # get sentiment scores from VADER
        sentiment_scores = self.analyzer.polarity_scores(text)
        return sentiment_scores
    
    def get_sentiment_label(self, compound_score):
        """
        Convert a compound sentiment score to a label.

        Args:
            compound_score: VADER compound score (-1 to 1)

        Returns:
            String sentiment label (positive, neutral, negative)
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    def analyze_dataframe(self, df, text_column='cleaned_text'):
        """
        Analyze sentiment for all texts in a DataFrame

        Args:
            df: Pandas DataFrame with text data
            text_column: Name of the column containing text to analyze

        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # create a copy
        result_df = df.copy()

        # apply sentiment analysis to each text
        sentiments = result_df[text_column].apply(self.analyze_text)

        # extract sentiment scores into separate columns
        result_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
        result_df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
        result_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
        result_df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])

        # add sentiment label
        result_df['sentiment_label'] = result_df['sentiment_compound'].apply(self.get_sentiment_label)

        return result_df
    
    def aggregate_sentiment_by_time(self, df, time_unit='D'):
        """
        Aggregate sentiment scores by time periods.

        Args:
            df: DataFrame with sentiment scores and datetime column
            time_unit: Time unit for grouping ('D' for data, 'H' for hour, etc.)

        Returns:
            DataFrame with aggregated sentiment by time period
        """
        if 'created_datetime' not in df.columns:
            raise ValueError("DataFrame must have 'created_datetime' column")
        
        # create time period column
        df_copy = df.copy()
        df_copy['time_period'] = df_copy['created_datetime'].dt.floor(time_unit)

        # group by time period and calculate average sentiment
        aggregated = df_copy.groupby('time_period').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_neutral': 'mean',
            'sentiment_negative': 'mean',
            'id': 'count'   # count of posts in the time period
        }).reset_index()

        # rename count column
        aggregated = aggregated.rename(columns={'id': 'post_count'})

        return aggregated
    
    def identify_trending_topics(self, df, n_topics=5):
        """
        Identify trending topics based on tokens and their average sentiment.
        
        Args:
            df: DataFrame with tokens and sentiment scores
            n_topics: Number of top topics to return
            
        Returns:
            DataFrame with top topics and their sentiment scores
        """
        if 'tokens' not in df.columns or 'sentiment_compound' not in df.columns:
            raise ValueError("DataFrame must have 'tokens' and 'sentiment_compound' columns")
        
        # explode tokens to get one row per token
        tokens_df = df.explode('tokens')

        # filter out tokens that are too short
        tokens_df = tokens_df[tokens_df['tokens'].str.len() > 2]

        # group by token and calculate metrics
        topics = tokens_df.groupby('tokens').agg({
            'sentiment_compound': 'mean',
            'id': 'count'
        }).reset_index()

        # rename columns
        topics = topics.rename(columns={
            'tokens': 'topic',
            'id': 'mention_count',
            'sentiment_compound': 'avg_sentiment'
        })

        # sort by mention count (descending) and get top N
        top_topics = topics.sort_values('mention_count', ascending=False).head(n_topics)

        # add sentiment label
        top_topics['sentiment_label'] = top_topics['avg_sentiment'].apply(self.get_sentiment_label)

        return top_topics
    
# example usage
if __name__ == "__main__":
    # example text
    texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The product is okay, nothing special."
    ]

    # create analyzer
    analyzer = SentimentAnalyzer()

    # analyze each text
    for text in texts:
        sentiment = analyzer.analyze_text(text)
        label = analyzer.get_sentiment_label(sentiment['compound'])
        print(f"Text: '{text}'")
        print(f"Sentiment: {label} (Compound: {sentiment['compound']:.3f})")
        print()