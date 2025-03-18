"""
Visualization Module

This module provides functions for creating visualizations for the sentiment analysis dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64


def create_sentiment_time_series(df, time_column='time_period'):
    """
    Create a time series chart of sentiment scores.
    
    Args:
        df: DataFrame with aggregated sentiment by time
        time_column: Name of the column containing time periods
        
    Returns:
        Plotly figure
    """
    # Check if required columns exist
    required_columns = [time_column, 'sentiment_compound', 'post_count']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must have columns: {required_columns}")
    
    # Create figure with secondary y-axis for post count
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=df[time_column],
            y=df['sentiment_compound'],
            name="Sentiment Score",
            line=dict(color='blue', width=3)
        ),
        secondary_y=False,
    )
    
    # Add post count bars
    fig.add_trace(
        go.Bar(
            x=df[time_column],
            y=df['post_count'],
            name="Post Count",
            marker_color='lightgray',
            opacity=0.7
        ),
        secondary_y=True,
    )
    
    # Set titles and labels
    fig.update_layout(
        title="Sentiment Score Over Time",
        xaxis_title="Time Period",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Sentiment Score (-1 to 1)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Posts", secondary_y=True)
    
    # Add a reference line at neutral sentiment (0)
    fig.add_shape(
        type="line",
        x0=df[time_column].min(),
        y0=0,
        x1=df[time_column].max(),
        y1=0,
        line=dict(color="red", width=1, dash="dash"),
        yref='y'
    )
    
    return fig


def create_sentiment_distribution_chart(df, sentiment_column='sentiment_label'):
    """
    Create a pie chart showing the distribution of sentiment labels.
    
    Args:
        df: DataFrame with sentiment labels
        sentiment_column: Name of the column containing sentiment labels
        
    Returns:
        Plotly figure
    """
    if sentiment_column not in df.columns:
        # Create a default pie chart showing a message
        fig = go.Figure(go.Pie(
            labels=["No sentiment data"],
            values=[1],
            textinfo="label"
        ))
        fig.update_layout(
            title="Sentiment Distribution (No Data Available)",
            annotations=[dict(
                text="Run sentiment analysis first",
                showarrow=False,
                font=dict(size=14)
            )]
        )
        return fig
    
    # Count occurrences of each sentiment label
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Define colors for each sentiment
    colors = {
        'positive': '#4CAF50',  # Green
        'neutral': '#FFC107',   # Amber
        'negative': '#F44336'   # Red
    }
    
    # Create pie chart
    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Sentiment Distribution',
        color='Sentiment',
        color_discrete_map=colors
    )
    
    # Update layout
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hole=0.3
    )
    
    return fig


def create_top_topics_chart(df, topic_column='topic', 
                        sentiment_column='avg_sentiment',
                        count_column='mention_count'):
    """
    Create a horizontal bar chart of top topics colored by sentiment.
    
    Args:
        df: DataFrame with topics and sentiment scores
        topic_column: Name of the column containing topics
        sentiment_column: Name of the column containing sentiment scores
        count_column: Name of the column containing mention counts
        
    Returns:
        Plotly figure
    """
    required_columns = [topic_column, sentiment_column, count_column]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must have columns: {required_columns}")
    
    # Sort by count (descending)
    df_sorted = df.sort_values(count_column, ascending=True)
    
    # Create a continuous color scale based on sentiment
    fig = px.bar(
        df_sorted,
        y=topic_column,
        x=count_column,
        color=sentiment_column,
        color_continuous_scale=['red', 'lightgray', 'green'],
        range_color=[-1, 1],
        title='Top Topics by Mention Count',
        orientation='h',
        labels={
            topic_column: 'Topic',
            count_column: 'Mention Count',
            sentiment_column: 'Sentiment Score'
        }
    )
    
    # Update layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_colorbar=dict(
            title='Sentiment',
            tickvals=[-1, 0, 1],
            ticktext=['Negative', 'Neutral', 'Positive']
        )
    )
    
    return fig


def generate_wordcloud(df, text_column='cleaned_text', sentiment_filter=None, 
                    sentiment_column='sentiment_label'):
    """
    Generate a word cloud from text data, optionally filtered by sentiment.
    
    Args:
        df: DataFrame with text data
        text_column: Name of the column containing text
        sentiment_filter: Optional filter for sentiment ('positive', 'neutral', 'negative')
        sentiment_column: Name of the column containing sentiment labels
        
    Returns:
        Base64 encoded image string
    """
    required_columns = [text_column]
    if sentiment_filter:
        required_columns.append(sentiment_column)
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must have columns: {required_columns}")
    
    # Filter by sentiment if specified
    if sentiment_filter:
        df = df[df[sentiment_column] == sentiment_filter]
    
    if df.empty:
        return None
    
    # Combine all text
    all_text = ' '.join(df[text_column].fillna('').astype(str))
    
    if not all_text.strip():
        return None
    
    # Define colors based on sentiment
    if sentiment_filter == 'positive':
        colormap = 'Greens'
    elif sentiment_filter == 'negative':
        colormap = 'Reds'
    elif sentiment_filter == 'neutral':
        colormap = 'Blues'
    else:
        colormap = 'viridis'
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(all_text)
    
    # Convert to image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save image to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64 string
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str