import tweepy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModel
import json
import time
from datetime import datetime, timedelta
import threading
import queue
import re
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class TwitterDataCollector:
    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        self.data_queue = queue.Queue()
        self.is_collecting = False

    def collect_tweets(self, keywords=None, duration_hours=1):
        if keywords is None:
            keywords = ["misinformation", "fake news", "conspiracy", "hoax", "propaganda",
                       "disinformation", "rumors", "false claim", "debunked", "fact check"]

        query = " OR ".join(f'"{k}"' if ' ' in k else k for k in keywords) + " -is:retweet lang:en"
        self.is_collecting = True
        end_time = datetime.now() + timedelta(hours=duration_hours)
        print(f"Starting tweet collection for {duration_hours} hour(s)...")
        print(f"Query: {query}")

        while self.is_collecting and datetime.now() < end_time:
            try:
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    tweet_fields=['author_id', 'created_at', 'public_metrics', 'context_annotations', 'conversation_id'],
                    expansions=['author_id'],
                    max_results=100
                ).flatten(limit=1000)

                for tweet in tweets:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'conversation_id': tweet.conversation_id
                    }
                    self.data_queue.put(tweet_data)
                time.sleep(60)
            except Exception as e:
                print(f"Error collecting tweets: {e}")
                time.sleep(30)

        self.is_collecting = False
        print("Tweet collection completed!")

    def get_user_network(self, user_id, max_followers=100):
        try:
            followers = self.client.get_users_followers(
                id=user_id,
                max_results=min(max_followers, 1000),
                user_fields=['verified', 'public_metrics', 'created_at']
            )
            network_data = []
            if followers.data:
                for follower in followers.data:
                    network_data.append({
                        'user_id': follower.id,
                        'username': follower.username,
                        'verified': follower.verified,
                        'followers_count': follower.public_metrics['followers_count'],
                        'created_at': follower.created_at
                    })
            return network_data
        except Exception as e:
            print(f"Error getting user network: {e}")
            return []

class DisinformationDetector:
    def __init__(self):
        self.classifier = pipeline("text-classification",
                                 model="martin-ha/toxic-comment-model",
                                 device=0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def extract_text_features(self, text):
        features = {}
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['mention_count'] = text.count('@')
        features['hashtag_count'] = text.count('#')
        try:
            toxicity_result = self.classifier(text)[0]
            features['toxicity_score'] = toxicity_result['score'] if toxicity_result['label'] == 'TOXIC' else 1 - toxicity_result['score']
        except:
            features['toxicity_score'] = 0.0
        return features

    def calculate_disinformation_risk(self, tweet_data):
        text = tweet_data['text']
        features = self.extract_text_features(text)
        risk_score = 0.0
        if features['exclamation_count'] > 2:
            risk_score += 0.1
        if features['caps_ratio'] > 0.3:
            risk_score += 0.15
        if features['toxicity_score'] > 0.5:
            risk_score += 0.2
        total_engagement = tweet_data['retweet_count'] + tweet_data['like_count'] + tweet_data['reply_count']
        if total_engagement > 1000:
            risk_score += 0.1
        if tweet_data['retweet_count'] > tweet_data['like_count'] * 0.5:
            risk_score += 0.15
        risk_score += features['url_count'] * 0.05
        return min(risk_score, 1.0)

class DisinformationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3):
        super(DisinformationGNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, concat=False))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

class NetworkAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.detector = DisinformationDetector()

    def build_propagation_graph(self, tweets_data):
        self.graph.clear()
        for tweet in tweets_data:
            tweet_id = tweet['id']
            author_id = tweet['author_id']
            self.graph.add_node(tweet_id, node_type='tweet', text=tweet['text'], created_at=tweet['created_at'], risk_score=self.detector.calculate_disinformation_risk(tweet))
            self.graph.add_node(author_id, node_type='user', tweet_count=1)
            self.graph.add_edge(author_id, tweet_id, edge_type='authored')
            if tweet['conversation_id'] and tweet['conversation_id'] != tweet_id:
                if tweet['conversation_id'] in [n for n in self.graph.nodes()]:
                    self.graph.add_edge(tweet_id, tweet['conversation_id'], edge_type='reply')

    def calculate_influence_metrics(self):
        metrics = {}
        if len(self.graph.nodes()) == 0:
            return metrics
        try:
            metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(self.graph, max_iter=1000)
            metrics['pagerank'] = nx.pagerank(self.graph)
        except:
            metrics['betweenness_centrality'] = {n: 0 for n in self.graph.nodes()}
            metrics['eigenvector_centrality'] = {n: 0 for n in self.graph.nodes()}
            metrics['pagerank'] = {n: 1/len(self.graph.nodes()) for n in self.graph.nodes()}
        return metrics

    def detect_communities(self):
        try:
            undirected_graph = self.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected_graph)
            return list(communities)
        except:
            return []

    def analyze_cascade_patterns(self):
        cascades = []
        tweet_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'tweet']
        for tweet_id in tweet_nodes:
            try:
                descendants = nx.descendants(self.graph, tweet_id)
                if len(descendants) > 0:
                    risk_score = self.graph.nodes[tweet_id].get('risk_score', 0)
                    cascades.append({
                        'source_tweet': tweet_id,
                        'cascade_size': len(descendants),
                        'risk_score': risk_score,
                        'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
                    })
            except:
                continue
        return sorted(cascades, key=lambda x: x['cascade_size'], reverse=True)

class VisualizationDashboard:
    def __init__(self):
        self.analyzer = NetworkAnalyzer()

    def create_network_visualization(self, graph):
        if len(graph.nodes()) == 0:
            return None
        pos = nx.spring_layout(graph, k=1, iterations=50)
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info = graph.nodes[node]
            if node_info.get('node_type') == 'tweet':
                risk_score = node_info.get('risk_score', 0)
                node_color.append(risk_score)
                node_text.append(f"Tweet: {node}<br>Risk: {risk_score:.2f}")
            else:
                node_color.append(0.5)
                node_text.append(f"User: {node}")
        edge_x, edge_y = [], []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                               marker=dict(showscale=True, colorscale='Reds', color=node_color, size=10,
                                           colorbar=dict(thickness=15, xanchor="left", titleside="right", title="Risk Score"))))
        fig.update_layout(title='Disinformation Propagation Network', showlegend=False, hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[ dict(text="Network showing information flow and risk levels", showarrow=False,
                                            xref="paper", yref="paper", x=0.005, y=-0.002)],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        return fig

    def create_risk_timeline(self, tweets_data):
        if not tweets_data:
            return None
        df = pd.DataFrame(tweets_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at')
        detector = DisinformationDetector()
        df['risk_score'] = df.apply(lambda row: detector.calculate_disinformation_risk(row), axis=1)
        fig = px.scatter(df, x='created_at', y='risk_score', size='retweet_count', color='risk_score',
                        color_continuous_scale='Reds', title='Disinformation Risk Over Time',
                        labels={'created_at': 'Time', 'risk_score': 'Risk Score'})
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        return fig

    def create_engagement_analysis(self, tweets_data):
        if not tweets_data:
            return None
        df = pd.DataFrame(tweets_data)
        detector = DisinformationDetector()
        df['risk_score'] = df.apply(lambda row: detector.calculate_disinformation_risk(row), axis=1)
        df['total_engagement'] = df['retweet_count'] + df['like_count'] + df['reply_count']
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Risk vs Engagement', 'Retweet Patterns',
                                         'Risk Distribution', 'Engagement Timeline'))
        fig.add_trace(go.Scatter(x=df['total_engagement'], y=df['risk_score'], mode='markers', name='Tweets',
                               marker=dict(color=df['risk_score'], colorscale='Reds')), row=1, col=1)
        fig.add_trace(go.Box(y=df['retweet_count'], name='Retweets'), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['risk_score'], name='Risk Distribution', marker_color='red', opacity=0.7),
                     row=2, col=1)
        df_sorted = df.sort_values('created_at')
        fig.add_trace(go.Scatter(x=list(range(len(df_sorted))), y=df_sorted['total_engagement'], mode='lines', name='Engagement'),
                     row=2, col=2)
        fig.update_layout(height=600, showlegend=False, title_text="Engagement and Risk Analysis")
        return fig

class RealTimeMonitor:
    def __init__(self, bearer_token):
        self.collector = TwitterDataCollector(bearer_token)
        self.analyzer = NetworkAnalyzer()
        self.dashboard = VisualizationDashboard()
        self.collected_tweets = []

    def start_monitoring(self, keywords=None, duration_hours=0.5):
        print("🚀 Starting Real-Time Disinformation Monitoring System")
        print("=" * 60)
        collection_thread = threading.Thread(target=self.collector.collect_tweets, args=(keywords, duration_hours))
        collection_thread.start()
        analysis_interval = 60
        last_analysis = time.time()
        while collection_thread.is_alive() or not self.collector.data_queue.empty():
            new_tweets = []
            while not self.collector.data_queue.empty():
                try:
                    tweet = self.collector.data_queue.get_nowait()
                    new_tweets.append(tweet)
                    self.collected_tweets.append(tweet)
                except queue.Empty:
                    break
            if new_tweets:
                print(f"📊 Collected {len(new_tweets)} new tweets. Total: {len(self.collected_tweets)}")
            if time.time() - last_analysis > analysis_interval and self.collected_tweets:
                self.perform_analysis()
                last_analysis = time.time()
            time.sleep(10)
        if self.collected_tweets:
            print("\n🔍 Performing final comprehensive analysis...")
            self.perform_analysis()
        collection_thread.join()

    def perform_analysis(self):
        print(f"\n🔬 ANALYSIS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        self.analyzer.build_propagation_graph(self.collected_tweets)
        metrics = self.analyzer.calculate_influence_metrics()
        cascades = self.analyzer.analyze_cascade_patterns()
        communities = self.analyzer.detect_communities()
        detector = DisinformationDetector()
        high_risk_tweets = []
        total_risk = 0
        for tweet in self.collected_tweets:
            risk = detector.calculate_disinformation_risk(tweet)
            total_risk += risk
            if risk > 0.6:
                high_risk_tweets.append({'id': tweet['id'], 'text': tweet['text'][:100] + '...',
                                         'risk': risk, 'engagement': tweet['retweet_count'] + tweet['like_count']})
        avg_risk = total_risk / len(self.collected_tweets) if self.collected_tweets else 0
        print(f"📈 THREAT ASSESSMENT:")
        print(f"   • Total tweets analyzed: {len(self.collected_tweets)}")
        print(f"   • Average risk score: {avg_risk:.3f}")
        print(f"   • High-risk tweets (>0.6): {len(high_risk_tweets)}")
        print(f"   • Network nodes: {self.analyzer.graph.number_of_nodes()}")
        print(f"   • Network edges: {self.analyzer.graph.number_of_edges()}")
        print(f"   • Detected communities: {len(communities)}")
        if high_risk_tweets:
            print(f"\n⚠️  TOP HIGH-RISK CONTENT:")
            for i, tweet in enumerate(sorted(high_risk_tweets, key=lambda x: x['risk'], reverse=True)[:3]):
                print(f"   {i+1}. Risk: {tweet['risk']:.3f} | Engagement: {tweet['engagement']}")
                print(f"      Text: {tweet['text']}")
        if cascades:
            print(f"\n🌊 PROPAGATION CASCADES:")
            for i, cascade in enumerate(cascades[:3]):
                print(f"   {i+1}. Size: {cascade['cascade_size']} | Risk: {cascade['risk_level']} ({cascade['risk_score']:.3f})")
        if metrics.get('pagerank'):
            top_influential = sorted(metrics['pagerank
