from flask import Flask, render_template, request
from flask_paginate import Pagination, get_page_parameter
import praw
import pandas as pd
import plotly.express as px
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

reddit = praw.Reddit(client_id='<>',
                     client_secret='<>',
                     user_agent='<>')

# Initialize the Hugging Face sentiment-analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def fetch_posts(subreddit_name, limit=100, filter_sentiment='all', search_keywords=''):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    users = {}
    for submission in subreddit.new(limit=limit):
        post_sentiment = sentiment_pipeline(submission.title)[0]['label']
        if (filter_sentiment == 'all' or post_sentiment.lower() == filter_sentiment.lower()) and (search_keywords.lower() in submission.title.lower()):
            posts.append({'title': submission.title, 'url': submission.url, 'sentiment': post_sentiment, 'author': submission.author.name})
            if submission.author.name in users:
                users[submission.author.name] += 1
            else:
                users[submission.author.name] = 1
    return posts, users

@app.route('/')
def home():
    posts, users = fetch_posts('TamilNadu')
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    offset = (page - 1) * per_page
    pagination_posts = posts[offset:offset + per_page]
    pagination = Pagination(page=page, total=len(posts), per_page=per_page, css_framework='bootstrap4')
    return render_template('index.html', posts=pagination_posts, pagination=pagination, users=users)

@app.route('/filter/<sentiment>')
def filter_posts(sentiment):
    posts, users = fetch_posts('TamilNadu', filter_sentiment=sentiment)
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10
    offset = (page - 1) * per_page
    pagination_posts = posts[offset:offset + per_page]
    pagination = Pagination(page=page, total=len(posts), per_page=per_page, css_framework='bootstrap4')
    return render_template('index.html', posts=pagination_posts, pagination=pagination, users=users)

@app.route('/search')
def search_posts():
    keywords = request.args.get('keywords', '')
    posts, users = fetch_posts('TamilNadu', search_keywords=keywords)
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
    posts, users = fetch_posts('TamilNadu')
    df = pd.DataFrame(posts)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title='Post Sentiment Distribution')

    # Calculate top 5 posters
    top_posters = sorted(users.items(), key=lambda x: x[1], reverse=True)[:5]

    # Find the Most Active Poster
    most_active_poster = max(users, key=users.get)

    # Find the Most Active Post
    most_active_post = max(posts, key=lambda x: x.get('num_comments', 0))

    return render_template('visualization.html', plot_html=fig.to_html(full_html=False), most_active_poster=most_active_poster, most_active_post=most_active_post, top_posters=top_posters)

@app.route('/network_analysis')
def network_analysis():
    subreddit_name = 'TamilNadu'
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
    plt.savefig('static/network_analysis.png')
    return render_template('network_analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
