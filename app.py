from flask import Flask, render_template, request
from flask_paginate import Pagination, get_page_parameter
import praw
import pandas as pd
import plotly.express as px
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

import base64
import numpy as np 
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pickle

app = Flask(__name__)

reddit = praw.Reddit(client_id='',
                     client_secret='',
                     user_agent='')

# Initialize the Hugging Face sentiment-analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')
subreddit_name='TamilNadu'
from transformers import pipeline
summarization_pipeline = pipeline('summarization')

# def summarize_text(text):
#     input_length = len(text.split())
#     max_length = min(150, input_length)  # Set max_length to input_length if input_length is less than 150
#     if max_length == 1:
#         return "" 
#     min_length = min(30, max_length - 10)  # Set min_length to a value less than max_length
#     return summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# def fetch_posts(subreddit_name, limit=100, filter_sentiment='all', search_keywords=''):
#     subreddit = reddit.subreddit(subreddit_name)
#     posts = []
#     users = {}
#     for submission in subreddit.new(limit=limit):
#         post_sentiment = sentiment_pipeline(submission.title)[0]['label']
#         if (filter_sentiment == 'all' or post_sentiment.lower() == filter_sentiment.lower()) and (search_keywords.lower() in submission.title.lower()):
#             # Calculate approximate upvotes and downvotes
#             upvotes = submission.upvote_ratio
#             downvotes = upvotes - submission.score

#             # Summarize post body
#             post_summary = summarize_text(submission.selftext) if submission.selftext else ""

#             # Get comments and summarize each comment
#             comments_summary = []
#             submission.comments.replace_more(limit=None)
#             for comment in submission.comments.list():
#                 if not isinstance(comment, praw.models.MoreComments) and comment.author:
#                     comment_summary = summarize_text(comment.body)
#                     comments_summary.append({
#                         'author': comment.author.name,
#                         'comment_summary': comment_summary
#                     })

#             posts.append({
#                 'title': submission.title,
#                 'url': submission.url,
#                 'sentiment': post_sentiment,
#                 'author': submission.author.name,
#                 'timestamp': submission.created_utc,
#                 'upvotes': upvotes,
#                 'downvotes': downvotes,
#                 'num_comments': submission.num_comments,
#                 'post_summary': post_summary,
#                 'comments_summary': comments_summary
#             })
#             if submission.author.name in users:
#                 users[submission.author.name] += 1
#             else:
#                 users[submission.author.name] = 1
#     return posts, users

def fetch_posts(subreddit_name, limit=1000, filter_sentiment='all', search_keywords=''):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    users = {}
    for submission in subreddit.new(limit=limit):
        post_sentiment = sentiment_pipeline(submission.title)[0]['label']
        if (filter_sentiment == 'all' or post_sentiment.lower() == filter_sentiment.lower()) and (search_keywords.lower() in submission.title.lower()):
            # Check if submission.author is None
            if submission.author is None:
                print("Submission author is None:", submission)
                continue  # Skip this submission and move to the next one

            # Calculate approximate upvotes and downvotes
            upvotes = submission.upvote_ratio 
            downvotes = 1 - upvotes

            posts.append({
                'title': submission.title,
                'url': submission.url,
                'sentiment': post_sentiment,
                'author': submission.author.name,
                'timestamp': submission.created_utc,
                'upvotes': upvotes,
                'downvotes': downvotes,
            })
            if submission.author.name in users:
                users[submission.author.name] += 1
            else:
                users[submission.author.name] = 1
    return posts, users


@app.route('/')
def home():
    posts, users = fetch_posts(subreddit_name)
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    offset = (page - 1) * per_page
    pagination_posts = posts[offset:offset + per_page]
    pagination = Pagination(page=page, total=len(posts), per_page=per_page, css_framework='bootstrap4')
    return render_template('index.html', posts=pagination_posts, pagination=pagination, users=users)

@app.route('/filter/<sentiment>')
def filter_posts(sentiment):
    posts, users = fetch_posts(subreddit_name, filter_sentiment=sentiment)
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    offset = (page - 1) * per_page
    pagination_posts = posts[offset:offset + per_page]
    pagination = Pagination(page=page, total=len(posts), per_page=per_page, css_framework='bootstrap4')
    return render_template('index.html', posts=pagination_posts, pagination=pagination, users=users)

@app.route('/search')
def search_posts():
    keywords = request.args.get('keywords', '')
    posts, users = fetch_posts(subreddit_name, search_keywords=keywords)
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    offset = (page - 1) * per_page
    pagination_posts = posts[offset:offset + per_page]
    pagination = Pagination(page=page, total=len(posts), per_page=per_page, css_framework='bootstrap4')
    return render_template('index.html', posts=pagination_posts, pagination=pagination, users=users)

@app.route('/post/<post_id>')
def post_detail(post_id):
    submission = reddit.submission(id=post_id)
    post_sentiment = sentiment_pipeline(submission.title)[0]['label']
    return render_template('post_detail.html', title=submission.title, url=submission.url, sentiment=post_sentiment)
  
@app.route('/visualization')
def visualization():
    posts, users = fetch_posts(subreddit_name)
    df = pd.DataFrame(posts)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title='Post Sentiment Distribution')

    # Calculate top 5 posters
    top_posters=[]
    top_posters = sorted(users.items(), key=lambda x: x[1], reverse=True)[:5]

    # Find the Most Active Poster
    most_active_poster = max(users, key=users.get)

    # Find the Most Active Post
    most_active_post = max(posts, key=lambda x: x.get('num_comments', 0))

    # Generate Word Cloud
    text = ' '.join(df['title'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Save Word Cloud image to a file
    wordcloud_path = 'D:/Collage/Real Time Ananlytics/DA3/Reddit Application/static/wordcloud.png'
    wordcloud.to_file(wordcloud_path)
    

    # Perform time series analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    daily_counts = df['date'].value_counts().sort_index()
    hourly_counts = df['timestamp'].dt.hour.value_counts().sort_index()

    # Create line charts
    fig_daily = px.line(x=daily_counts.index, y=daily_counts.values, labels={'x':'Date', 'y':'Number of Posts'}, title='Daily Posting Activity')
    fig_hourly = px.line(x=hourly_counts.index, y=hourly_counts.values, labels={'x':'Hour of the Day', 'y':'Number of Posts'}, title='Hourly Posting Activity')

    #Heatmap 
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # Convert 'date_interval' from Period to string for JSON serialization
    df['date_interval'] = df['timestamp'].dt.to_period('D').astype(str)

    # Now you can group by 'date_interval' and create your heatmap as before
    frequency = df.groupby('date_interval').size().reset_index(name='post_count')
    heatmap_data = frequency.pivot(index='date_interval', columns='post_count', values='post_count')
    fig_heatmap = px.imshow(heatmap_data, color_continuous_scale='YlOrRd', labels=dict(x="Date Interval", y="Number of Posts", color="Post Count"))
    fig_heatmap.update_layout(title='Post Frequency Heatmap')
    return render_template('visualization.html', plot_html=fig.to_html(full_html=False), most_active_poster=most_active_poster, most_active_post=most_active_post, top_posters=top_posters, fig_daily=fig_daily.to_html(full_html=False), fig_hourly=fig_hourly.to_html(full_html=False), fig_heatmap=fig_heatmap.to_html(full_html=False))
@app.route('/network_analysis')
def network_analysis():
    limit = 10  # Limit the number of submissions to analyze
    subreddit = reddit.subreddit(subreddit_name)

    G = nx.Graph()

    # Analyze comments
    for submission in subreddit.new(limit=limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if not isinstance(comment, praw.models.MoreComments) and comment.author and submission.author:
                # Add an edge between the submission author and comment author
                G.add_edge(submission.author.name, comment.author.name)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='skyblue', edge_color='gray', arrowsize=20)
    plt.savefig('D:/Collage/Real Time Ananlytics/DA3/Reddit Application/static/network_analysis.png')
    return render_template('network_analysis.html')

@app.route('/topics')
def topics():
    posts, users = fetch_posts(subreddit_name)
    texts = [[word for word in post['title'].lower().split()] for post in posts]  # Extract title text

    # Check if texts is empty or not
    if not texts:
        return "No texts found. Please check your data."

    # Create a dictionary representation of the documents
    dictionary = Dictionary(texts)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Check the dictionary size after filtering extremes
    print("Dictionary size after filtering extremes:", len(dictionary))

    # Bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Apply LDA
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    # Visualize topics
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    
    return render_template('topics.html', vis=vis, coherence=coherence_lda)
# Flask application
@app.route('/save_csv')
def save_csv():
    posts, _ = fetch_posts(subreddit_name)
    df = pd.DataFrame(posts)
    csv_path = 'D:/Collage/Real Time Ananlytics/DA3/Reddit Application/static/reddit_posts.csv'
    df.to_csv(csv_path, index=False)
    return f'CSV file saved at {csv_path}'

if __name__ == '__main__':
    app.run(debug=True)
