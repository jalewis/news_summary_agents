#!/usr/bin/env python3

# app.py

# Suppress Pydantic V2 warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import yaml
import logging
import requests
import feedparser
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# Using standard LangChain modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# For embeddings and similarity checks
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

import smtplib
from email.message import EmailMessage
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import List, Dict
from datetime import date
import os

# If you still use these tools:
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool


###############################################################################
# Data Models
###############################################################################
class ArticleSummary(BaseModel):
    title: str = Field(..., description="Title of the Article")
    link: str = Field(..., description="URL of the Article")
    summary: str = Field(..., description="Summary of the article")
    usefulness: float = Field(..., description="Relevance and informational value of the article, range between 0 and 1")
    urgency: float = Field(..., description="Urgency of the news in the article, range between 0 and 1")

class ArticleSummaries(RootModel):
    root: List[ArticleSummary] = Field(..., description="A list of article summaries")


###############################################################################
# Config & Logging
###############################################################################
def load_config(config_path):
    try:
        with open(os.path.join('config', config_path), 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise

def load_agent_config(config_path):
    try:
        with open(os.path.join('config', config_path), 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading agent config from {config_path}: {e}")
        raise

def load_task_config(config_path):
    try:
        with open(os.path.join('config', config_path), 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading task config from {config_path}: {e}")
        raise

def setup_logging(log_level_name):
    level = getattr(logging, log_level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level_name}")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )


###############################################################################
# Web & RSS helpers
###############################################################################
def fetch_and_parse_webpage(url, max_retries):
    session = requests.Session()
    retry = Retry(
        connect=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        response = session.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        if not text:
            logging.warning(f"No text content found for URL {url}")
            return None
        return text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching or parsing URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing URL {url}: {e}")
        return None

def fetch_and_parse_rss(rss_urls, keywords, max_retries=3):
    all_articles = []
    for url in rss_urls:
        try:
            logging.info(f"Fetching RSS feed: {url}")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article_text = fetch_and_parse_webpage(entry.link, max_retries)
                if article_text and any(k.lower() in article_text.lower() for k in keywords):
                    all_articles.append({
                        'title': entry.title,
                        'link': entry.link,
                        'content': article_text
                    })
                    logging.info(f"RSS article added: {entry.title}")
        except Exception as e:
            logging.error(f"Error processing RSS feed {url}: {e}")
    return all_articles


###############################################################################
# Article deduplication
###############################################################################
def deduplicate_articles(articles, embedding_model, threshold=0.80):
    """
    Generate embeddings for each article, then remove duplicates based on a
    cosine similarity threshold.
    """
    logging.info(f"Deduplicating articles")
    try:
        model = SentenceTransformer(embedding_model)
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        logging.info(f"Attempting to use all-MiniLM-L6-v2 instead")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    if not articles:
        logging.warning(f"No articles to deduplicate")
        return []

    embeddings = []
    for article in articles:
        embeddings.append(model.encode(article['content']))

    filtered_articles = [articles[0]]
    filtered_embeddings = [embeddings[0]]

    for i in range(1, len(articles)):
        is_duplicate = False
        for j in range(len(filtered_embeddings)):
            similarity = 1 - cosine(embeddings[i], filtered_embeddings[j])
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_articles.append(articles[i])
            filtered_embeddings.append(embeddings[i])
    return filtered_articles


###############################################################################
# Email
###############################################################################
def send_email(smtp_server, smtp_port, from_address, to_addresses, subject, email_body):
    message = EmailMessage()
    message.set_content(email_body)
    message['Subject'] = subject
    message['From'] = from_address
    message['To'] = ", ".join(to_addresses)

    logging.info(f"Preparing email to: {message['To']}")
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.send_message(message)
            logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Error sending email: {e}")


###############################################################################
# Ranking
###############################################################################
def rank_articles(articles, ranking_agent):
    ranked_articles = []
    for article in articles:
        task = Task(
            description=(
                "Rank this article from 0 to 1 on usefulness and 0 to 1 on urgency, "
                "0 being not useful or urgent and 1 being very useful and urgent: "
                f"{article['content']}"
            ),
            agent=ranking_agent,
            expected_output="A summary with usefulness and urgency scores"
        )
        try:
            crew = Crew(
                agents=[ranking_agent],
                tasks=[task],
                process=Process.sequential
            )
            result = crew.kickoff()
            ranked_articles.append(result)
        except ValidationError as e:
            logging.error(f"Error validating rank_article output: {e}")
            continue
    return ranked_articles


###############################################################################
# (Optional) LLM Factory
###############################################################################
class LLMFactory:
    @staticmethod
    def create_llm(config=None):
        """
        Example factory if you want a separate method for building your LLM.
        """
        try:
            return OllamaLLM(
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://192.168.120.141:11434'),
                model=os.getenv('OLLAMA_MODEL', 'research-mistral-nemo:latest'),
                temperature=0.1,
            )
        except Exception as e:
            logging.error(f"LLM initialization error: {e}")
            raise


###############################################################################
# Main Execution
###############################################################################
def main():
    # Load configurations
    config = load_config('config.yaml')
    setup_logging(config['log_level'])
    agent_config = load_agent_config('agents.yaml')
    task_config = load_task_config('tasks.yaml')

    # Ollama config
    ollama_llm_model = config['ollama']['llm_model']
    ollama_embed = config['ollama']['embedding_model']
    ollama_base_url = config['ollama']['base_url']

    # Email configuration
    email_config = config['email']
    smtp_server = email_config['smtp_server']
    smtp_port = email_config['smtp_port']
    from_address = email_config['from_address']
    to_addresses = email_config['to_addresses']
    subject = f"{email_config['subject']} {date.today().strftime('%Y-%m-%d')}"

    # 1. Initialize LLM with the new langchain-ollama class
    llm = LLM(
        model=f"ollama/{ollama_llm_model}",      # The model name
        base_url=ollama_base_url,    # Where your Ollama server is running
        temperature=0.1
    )

    # 2. Create Agents
    researcher = Agent(
        role='Research Analyst',
        goal='Provide up-to-date market analysis',
        backstory='An expert analyst with a keen eye for market trends.',
        tools=[
            SerperDevTool(api_key=config['search_engines']['serp']['api_key'])
            if config['search_engines']['serp'].get('enabled') else None
        ],
        llm=llm,
        verbose=True
    )

    ranking_agent = Agent(
        role='Content Ranker',
        goal='Evaluate and rank content based on relevance and urgency',
        backstory='A seasoned content evaluator with expertise in identifying valuable information.',
        llm=llm,
        verbose=True
    )

    summarizer_agent = Agent(
        role='Content Summarizer',
        goal='Create concise and informative summaries',
        backstory='An expert in distilling complex information into clear summaries.',
        llm=llm,
        verbose=True
    )

    global_summarizer_agent = Agent(
        role='Global Content Analyst',
        goal='Synthesize information across multiple sources',
        backstory='A specialist in connecting dots and identifying broader patterns.',
        llm=llm,
        verbose=True
    )

    # 3. Process Topics
    all_ranked_summarized_articles = {}
    for topic_data in config['task']['top_topics']:
        topic = topic_data['topic']
        keywords = topic_data['keywords']
        all_articles = []

        # 3a. Fetch RSS articles
        rss_articles = fetch_and_parse_rss(config['task']['rss_feeds'], keywords, max_retries=3)
        all_articles.extend(rss_articles)

        # 3b. Deduplicate
        deduplicated_articles = deduplicate_articles(all_articles, ollama_embed)

        # 3c. Agent: Research (Search Engine)
        research_task = Task(
            description=task_config['research_task']['description'].format(
                topic=topic,
                keywords=' '.join(keywords)
            ),
            agent=researcher,
            expected_output="A list of URLs based on the search query"
        )

        try:
            research_crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                process=Process.sequential
            )
            search_results = research_crew.kickoff()
            
            if search_results and hasattr(search_results, 'text'):
                 
                #split only if not None or empty string
                if search_results.text and search_results.text != "":
                   
                   for url in search_results.text.split("\n"):
                       #perform a basic validation to ensure the text is a valid URL
                       if url.startswith('http://') or url.startswith('https://'):
                           text = fetch_and_parse_webpage(url, max_retries=3)
                           if text:
                               deduplicated_articles.append({
                                'title': url,
                                   'link': url,
                                   'content': text
                               })
                               logging.info(f"Search engine article added: {url}")
                       else:
                            logging.warning(f"Invalid URL detected: {url}")
        except Exception as e:
            logging.error(f"Error during research task: {e}")

        # 3d. Agent: Rank
        ranked_articles = rank_articles(deduplicated_articles, ranking_agent)

        # 3e. Agent: Summarize
        summarized_articles = []
        for article in ranked_articles:
            summarize_task = Task(
                description=task_config['summarize_article_task']['description'].format(
                    article_content=article.summary
                ),
                agent=summarizer_agent,
                expected_output="A summarized version of the input article"
            )

            try:
                summarize_crew = Crew(
                    agents=[summarizer_agent],
                    tasks=[summarize_task],
                    process=Process.sequential
                )
                result = summarize_crew.kickoff()

                # Copy over the usefulness and urgency from the ranking step
                result.usefulness = article.usefulness
                result.urgency = article.urgency

                summarized_articles.append(result)
            except ValidationError as e:
                logging.error(f"Error validating task output: {e}")
                continue

        # Sort articles by combined (usefulness + urgency) desc
        all_ranked_summarized_articles[topic] = sorted(
            summarized_articles,
            key=lambda x: (x.usefulness + x.urgency),
            reverse=True
        )

    # 4. Create Global Summaries
    global_summaries = {}
    for topic, summarized_articles in all_ranked_summarized_articles.items():
        global_summary_task = Task(
            description=task_config['global_summary_task']['description'].format(
                summarized_articles=summarized_articles
            ),
            agent=global_summarizer_agent,
            expected_output="A comprehensive summary"
        )

        global_summary_crew = Crew(
            agents=[global_summarizer_agent],
            tasks=[global_summary_task],
            process=Process.sequential
        )
        global_summaries[topic] = global_summary_crew.kickoff()

    # 5. Send an email with everything
    email_body = ""
    for topic, summaries in global_summaries.items():
        email_body += f"Topic: {topic}\n\n"
        email_body += f"Overall Summary: {summaries}\n\n"
        for article in all_ranked_summarized_articles[topic]:
            email_body += f"Article: {article.title}\n"
            email_body += f"Summary: {article.summary}\n"
            email_body += f"URL: {article.link}\n"
            email_body += f"Usefulness: {article.usefulness}\n"
            email_body += f"Urgency: {article.urgency}\n\n"

    send_email(smtp_server, smtp_port, from_address, to_addresses, subject, email_body)


if __name__ == "__main__":
    main()