"""
Reddit Data Collection Script

This script collects posts from Reddit based on search terms and saves them
to CSV files for further processing.
"""

import os
import csv
import json
import datetime
import praw
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# directory for storing raw data
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def initialize_reddit_client():
    """
    Initialize and return a Reddit API client using credentials from environment variables.
    """
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
        )
        print("Successfully connected to Reddit API")
        return reddit
    except Exception as e:
        print(f"Error connecting to Reddit API: {e}")
        return None
    
def collect_subreddit_posts(reddit, subreddit_name, limit=100):
    """
    Collect posts from a specific subreddit.
    
    Args:
        reddit: Initialized Reddit API client
        subreddit_name: Name of the subreddit to collect from
        limit: Maximum number of posts to collect
        
    Returns:
        List of collected posts with relevant data
    """
    collected_posts = []

    try:
        subreddit = reddit.subreddit(subreddit_name)

        # collect hot posts
        for post in subreddit.hot(limit=limit):
            post_data = {
                'id': post.id,
                'title': post.title,
                'body': post.selftext,
                'created_utc': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments,
                'subreddit': subreddit_name,
                'permalink': post.permalink,
                'url': post.url,
                'collected_at': datetime.datetime.now().isoformat()
            }
            collected_posts.append(post_data)

        print(f"Collected {len(collected_posts)} posts from r/{subreddit_name}")
        return collected_posts
    
    except Exception as e:
        print(f"Error collecting posts from r/{subreddit_name}: {e}")
        return []
    
def collect_posts_by_search(reddit, search_term, limit=100, sort='relevance'):
    """
    Collect posts based on a search term across Reddit.
    
    Args:
        reddit: Initialized Reddit API client
        search_term: Term to search for
        limit: Maximum number of posts to collect
        sort: Sorting method ('relevance', 'hot', 'new', 'top', 'comments')
        
    Returns:
        List of collected posts with relevant data"
    """
    collected_posts = []
    
    try:
        # search for posts containing the search term
        for post in reddit.subreddit('all').search(search_term, limit=limit, sort=sort):
            post_data = {
                'id': post.id,
                'title': post.title,
                'body': post.selftext,
                'created_utc': post.created_utc,
                'score': post.score,
                'num_comments': post.num_comments,
                'subreddit': post.subreddit.display_name,
                'permalink': post.permalink,
                'url': post.url,
                'search_term': search_term,
                'collected_at': datetime.datetime.now().isoformat()
            }
            collected_posts.append(post_data)
        
        print(f"Collected {len(collected_posts)} posts related to '{search_term}'")
        return collected_posts
    
    except Exception as e:
        print(f"Error collecting posts for search term '{search_term}': {e}")
        return []
    
def save_posts_to_csv(posts, filename):
    """
    Save collected posts to a CSV file.
    
    Args:
        posts: List of post dictionaries
        filename: Name of the CSV file to save to
    """
    if not posts:
        print("No posts to save")
        return
    
    filepath = os.path.join(RAW_DATA_DIR, filename)

    try:
        # extract fieldnames from the first post
        fieldnames = posts[0].keys()

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for post in posts:
                writer.writerow(post)

        print(f"Saved {len(posts)} posts to {filepath}")

    except Exception as e:
        print(f"Error saving posts to CSV: {e}")

def main():
    """
    Main function to collect Reddit data based on command line arguments.
    """
    # check if environment variables are set
    if not all([os.getenv('REDDIT_CLIENT_ID'), 
                os.getenv('REDDIT_CLIENT_SECRET'), 
                os.getenv('REDDIT_USER_AGENT')]):
        print("Error: Reddit API credentials not found in .env file")
        print("Please create a .env file with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
        return
    
    # initialize Reddit client
    reddit = initialize_reddit_client()
    if not reddit:
        return
    
    # define your search terms or subreddits here
    search_terms = ['artificial intelligence', 'machine learning']
    subreddits = ['datascience', 'machinelearning']

    # collect posts by search terms
    for term in search_terms:
        # replace spaces with underscores for filename
        term_filename = term.replace(' ', '_')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"search_{term_filename}_{timestamp}.csv"

        posts = collect_posts_by_search(reddit, term, limit=100)
        save_posts_to_csv(posts, filename)

if __name__ == "__main__":
    main()