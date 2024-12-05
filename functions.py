import os
import spacy
import string
import tqdm
import csv
from gensim.models import Word2Vec
import gensim
from collections import defaultdict
import math
import pandas as pd
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import tempfile
import seaborn as sns
from scipy.spatial.distance import cosine
from wordfreq import top_n_list
from wordcloud import WordCloud

# This function cleans the text, tokenizes and lemmatizes
def preprocess_text(text, lemmatize=True, remove_stopwords=False):
    nlp = spacy.load("es_core_news_sm")
    nlp.max_length = 2000000
    remove_chars = string.punctuation + '1234567890“”—�–\n-:º°»«ª´·―…▪'
    table = str.maketrans(remove_chars, ' ' * len(remove_chars))
    text = text.translate(table)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('’', '').replace('‘', '')
    
    doc = nlp(text.lower())
    
    if lemmatize:
        if remove_stopwords:
            return [token.lemma_ for token in doc if not token.is_stop and token.lemma_]
        return [token.lemma_ for token in doc if token.lemma_]
    
# We make a tfidf function that takes this specific structure of list we made and some interval and returns the tfidf vectors for the laws in that time period selected, along with the vocabulary index, in order to later recognize which words belong to which index
def tfidf(data, interval = None):
    new_data = []
    if interval:
        new_data = [(law, year, words) for law, year, words in data if year in interval]
    else:
        new_data = data
    
    idf_counts = defaultdict(int)
    for _, _, words in new_data: 
        vocab = set(words) 
        for word in vocab: 
            idf_counts[word] += 1 
    
    idf_values = defaultdict(float)
    num_documents = len(interval)
    for word in idf_counts: 
        idf_values[word] = math.log(num_documents / idf_counts[word]) 

    tf_counts = defaultdict(lambda: defaultdict(int))
    doc_lengths = defaultdict(int)
    for _, year, words in new_data:  
        for word in words:
            tf_counts[year][word] += 1 
        doc_lengths[year] += len(words)
    
    tf_value = defaultdict(lambda: defaultdict(float))
    for _, year, words in new_data:
        for word in tf_counts[year]: 
            tf_value[year][word] = tf_counts[year][word] / doc_lengths[year]

    tfidf_vectors = defaultdict(list)
    vocab = sorted(idf_values.keys())
    vocab_index = {word: idx for idx, word in enumerate(vocab)} 

    for year in tf_value: 
        for word in vocab:
            if word in tf_value[year]:
                tfidf_score = tf_value[year][word]*idf_values[word]
            else:
                tfidf_score = 0
            tfidf_vectors[year].append(tfidf_score)
    
    return tfidf_vectors, vocab_index

def get_decade(year):
    return (year // 10) * 10

def random_tfidf(data, seed, bootstrap_iterations = 100, intervals = None):
    if not intervals:
        intervals = make_intervals()        

    laws_by_decade = defaultdict(list)
    for law, year, words in data:
        year = int(year)
        for interval in intervals:
            if year in interval:
                laws_by_decade[f'{interval[0]}-{interval[-1]}'].append((law, year, words))
    
    bootstrap_results = defaultdict(pd.DataFrame)
    for iter in tqdm.tqdm(range(bootstrap_iterations)):
        selected_laws = []
        random.seed(seed + iter)
        for decade, laws in laws_by_decade.items():
            selected_laws.extend(random.sample(laws, 50))
        vectors = defaultdict(pd.DataFrame)
        for interval in intervals:
            tfidf_vectors, vocab_index = tfidf(selected_laws, interval=interval)
            tfidf_vectors = pd.DataFrame(tfidf_vectors, index=vocab_index.keys())
            vectors[f'{interval[0]}-{interval[-1]}'] = tfidf_vectors
        
        preliminary = defaultdict(lambda: defaultdict(float))
        for decade in vectors:
            tfidf_vectors = vectors[decade]
            tfidf_means = tfidf_vectors.mean(axis=1)
            top_words = tfidf_means.nlargest(300)
            for word, value in top_words.items():
                preliminary[decade][word] = value
        preliminary = pd.DataFrame(preliminary)
        
        bootstrap_results[iter] = preliminary
    
    common_words = set.intersection(
        *[set(df.index) for df in bootstrap_results.values()]
    )
    
    results = []
    for word in common_words:
        if len(word) > 1:
            for decade in laws_by_decade.keys():
                values = [df.loc[word, decade] for df in bootstrap_results.values()]
                mean_value = pd.Series(values).mean()
                results.append({"word": word, "decade": decade, "value": mean_value})

    final_df = pd.DataFrame(results).pivot(index="word", columns=["decade"], values="value")    
    
    return final_df

def words_intervals(data, intervals = None):
    if not intervals:
        intervals = make_intervals() 
    words_by_interval = defaultdict(list)
    for interval in intervals:
        start_year = interval[0]
        end_year = interval[-1]
        for element in data:
            _, year, words = element 
            year = int(year)
            if start_year <= year <= end_year:
                words_by_interval[f"{start_year}_{end_year}"].append(words)
    
    return words_by_interval


# This function searches for the words that are common to the vocabulary of two Word2Vec models, and later returns those that are "most common", according to the wordfreq module.
# We do this in order to later project one model into another.
# We use the "most common" words in our mapping because we consider that these are the words that are the least likely to change meaning through time
def get_mapping(model_1, model_2):
    vocab_1 = set(list(model_1.wv.index_to_key))
    vocab_2 = set(list(model_2.wv.index_to_key))
    
    mapping = list(vocab_1 & vocab_2)
    return [word for word in mapping if word in top_n_list('es', 500) and len(word) > 1]

# These function takes two models, and a list of words, projects one space into the other and for each of the words given, find the most similar words across projections. 
# Also measures the semantic drift for these words across time
def projection(model_1, model_2):
    mapping = get_mapping(model_1=model_1, model_2=model_2)
    A, B = [], []
    
    for word in mapping:
        if word in model_1.wv and word in model_2.wv:
            A.append(model_1.wv[word])
            B.append(model_2.wv[word])
    
    A, B = np.array(A), np.array(B)
    BtA = np.dot(B.T, A)
    U, S, Vt = np.linalg.svd(BtA)
    R = np.dot(U, Vt)
    return R
 
def similar_words_df(model_1, model_2, words):
    R = projection(model_1 = model_1, model_2 = model_2)
    proj = []
    drift_data = []
    for word in words:
        if word in model_1.wv:
            v = model_1.wv.get_vector(word)
            f = np.dot(v, R)
            similar_words = model_2.wv.similar_by_vector(f, topn=10)

            for similar_word, score in similar_words:
                proj.append({
                    "word": word,
                    "similar_word": similar_word,
                    "score": score
                })
            
            drift_distance = 1 - cosine(v, f)
            drift_data.append({
                "word": word,
                "semantic_drift": drift_distance
            })
    
    return proj, drift_data

def projected_vector(model_1, model_2, word):
    R = projection(model_1 = model_1, model_2 = model_2)
    if word in model_1.wv:
        v = model_1.wv.get_vector(word)
        f = np.dot(v, R)
    
    return f

def semantic_drift_time(data, nombre):
	plt.figure(figsize=(12, 6))
	sns.lineplot(
		data= data,
		x="target_model",
		y="semantic_drift",
		hue="word",
		marker="o"
	)

	plt.xlabel("Interval", fontsize=12)
	plt.ylabel("Cosine Distance", fontsize=12)
	plt.xticks(rotation=45)
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()

	plt.savefig(f'./output/semantic_drift_{nombre}.png')
	plt.show()
 
def get_interval(year, intervals):
    for label, years in intervals.items():
        if int(year) in years:
            return label
    return 'Unknown'

def make_intervals():
    all_years = list(range(1890, 2025))
    intervals = []
    for i in range(10, len(all_years), 10):
        if i < 20:
            start = 1890
            end = 1909
        else:
            start = all_years[i]
            end = all_years[i + 9] if i + 9 < len(all_years) else all_years[-1]
        interval = list(range(start, end + 1, 1))
        intervals.append(interval)
    return intervals

def extract_years(model_name):
    match = re.search(r'model_(\d{4})_(\d{4})', model_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return model_name  

def get_period_from_model(model_name, intervals):
    period_name = model_name.replace('model_', '')
    return period_name