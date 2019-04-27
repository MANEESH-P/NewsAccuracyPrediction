import requests
import re
import validators
import feedparser
import csv
from bs4 import BeautifulSoup
from eventregistry import *
import pandas as pd
from prediction import *
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions, MetadataOptions
from rake_nltk import Rake
from flask import Flask, render_template, request
from rep import mlToOut

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


# INIT ALL ML
print("loading tensorflow  model")
sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = loadML()


@app.route("/")
@app.route("/home")
def home():

    return render_template('home.html')


@app.route("/scrape", methods=["POST"])
def scrape():
    news_articles = []
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2018-11-16',
        iam_apikey='4kxtefSt-VgDt3LGbteO7tv0eAczVWdvXJcMIKhHdJfo',
        url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
    )

    if(request.method == "POST"):

        # response = natural_language_understanding.analyze(
        #     url=request.form["keyword"],
        #     features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=4))).get_result()
        # phrases = [request.form["keyword"]]
        # keywords = []
        # for i in range(len(phrases)):
        #     keywords = keywords + phrases[i].split()
        # phrases = phrases[0].split()
        if(validators.url(request.form["keyword"])):
            response = natural_language_understanding.analyze(
                url=request.form["keyword"],
                features=Features(metadata=MetadataOptions())).get_result()

            article_title = response["metadata"]["title"]

            response2 = natural_language_understanding.analyze(
                text=article_title,
                features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=15))).get_result()

            keywords = []
            for keyword in response2['keywords']:
                if keyword['relevance'] > 0.5 and len(keywords) < 8:
                    keywords.append(keyword['text'])

            new_keywords = []
            for i in range(len(keywords)):
                new_keywords = new_keywords + keywords[i].split()
        else:
            phrases = [request.form["keyword"]]
            new_keywords = []
            new_keywords = phrases[0].split()

        # r = Rake()
        # r.extract_keywords_from_text(article_title)
        # phrases = r.get_ranked_phrases()
        # print(phrases)
        # new_phrases = []
        # for i in range(len(phrases)):
        #     new_phrases = new_phrases + phrases[i].split()
        # new_phrases = phrases[:15]
        # for i in range(len(phrases)):
        #     if(len(phrases[i]) >= 3):
        #         new_phrases = new_phrases + phrases[i].split()
        #     else:
        #         new_phrases = new_phrases + phrases[i]
        # article_title = response["metadata"]["title"]
        # print(article_title)
        # new_phrases = article_title.split()
        # for i in range(len(response["keywords"])):
        #     if(response["keywords"][i]["relevance"] >= 0.4):
        #         new_phrases = new_phrases + [response["keywords"][i]["text"]]
        # keywords = []
        # for keyword in response['keywords']:
        #     keywords.append(keyword['text'])
        # new_phrases = keywords

    print(new_keywords)

    print('type of phrases:')
    print(type(new_keywords))
    api_key = 'e7c28375-a0b6-4566-b1c4-36b2af9f0009'
    er = EventRegistry(apiKey=api_key)

    q = QueryArticlesIter(
        keywords=new_keywords,
        keywordsLoc="title",
        sourceUri=QueryItems.OR(['indianexpress.com',
                                 'thehindu.com', 'news18.com',
                                 'timesofindia.indiatimes.com', 'firstpost.com', 'deccanchronicle.com'
                                 ]))

    csv_columns = ['uri', 'lang', 'isDuplicate', 'date', 'time', 'dateTime', 'dataType', 'sim',
                   'url', 'title', 'body', 'source', 'authors', 'image', 'eventUri', 'sentiment', 'wgt']
    csv_file = "test.csv"

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for article in q.execQuery(er, sortBy="rel", maxItems=500):
                # print(article)
                news_articles.append(article)
                writer.writerow(article)
    except IOError:
        print("I/O error")

    #checking if related news articles count is zero
    if(len(news_articles)==0):
        return render_template('noArticlesFound.html',news_articles=news_articles)
    else:
        Body_ID = []
        Id = []
        f = pd.read_csv("test.csv")

        # taking column title from test.csv and creating test_stances_unlabeled.csv
        keep_col = ['title']
        new_f = f[keep_col]
        new_f = new_f.rename(columns={'title': 'Headline'})
        for i in range(new_f['Headline'].count()):
            Body_ID.append(i+1)
        new_f['Body ID'] = Body_ID
        new_f.to_csv("test_stances_unlabeled.csv", index=False)

        # taking column body from test.csv and creating test_bodies.csv
        keep_col = ['body']
        idx = 0
        new_f = f[keep_col]
        new_f = new_f.rename(columns={'body': 'articleBody'})
        new_f.insert(loc=idx, column='Body ID', value=Body_ID)
        new_f.to_csv("test_bodies.csv", index=False)

        # taking column url from test.csv and creating url.csv
        keep_col = ['url']
        idx = 0
        new_f = f[keep_col]
        for i in range(new_f['url'].count()):
            Id.append(i)
        new_f.insert(loc=idx, column='id', value=Id)
        new_f.to_csv("url.csv", index=False)

        newsData = pd.read_csv('url.csv')
        URLs = newsData['url'].tolist()
        SourceName = []
        BodyID = newsData['id'].tolist()

        for news in news_articles:
            SourceName.append(news['source']['title'])

        # newsData = pd.read_csv('url.csv')
        # URLs = newsData['url'].tolist()
        # SourceName = newsData['source'].tolist()
        # BodyID = newsData['id'].tolist()

        Stances = runModel(sess, keep_prob_pl, predict, features_pl,
                        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

        BodyID = range(len(Stances))
        print("length of url array: ", len(URLs))
        print("length of BodyId array: ", len(BodyID))
        print("length of Stances array: ", len(Stances))
        print("length of SourceName array: ", len(SourceName))
        ml_output = pd.DataFrame(
            {'BodyID': BodyID,
                'Stances': Stances,
                'SourceName': SourceName,
                'URL': URLs
            })

        response = ml_output.reset_index(drop=True)
        response = response.to_dict(orient='records')
        final_score = mlToOut.returnOutput(ml_output)
        final_score = (final_score + 1)/2
        print("final score: ", final_score)
        print("length of stances", len(Stances))
        print("Stances: ", Stances)
        new_data = {'news_articles':news_articles,'final_score':final_score}

        return render_template('downloading.html', data=new_data)
        # return render_template('home.html', final_score=final_score)


@app.route("/stances", methods=["POST"])
def stance():
    colnames = ['stance']
    data = pd.read_csv('predictions_test.csv', names=colnames)
    stances = data.stance.tolist()
    f = pd.read_csv("test.csv")
    news_titles = f['title'].tolist()
    new_data = {'title': news_titles, 'opinion': stances}
    return render_template('stances.html', data=new_data)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = pd.read_csv('predictions_test.csv')
    stance = file['Stance'].value_counts()
    dictionary = {}
    for key in stance.keys():
        dictionary.update({key: stance[key]})
    stance = []
    total_count = 0
    for value in dictionary.values():
        total_count = total_count + value
    for key in dictionary.keys():
        stance.append({'stance': key, 'count': round(
            (dictionary[key]/total_count)*100, 2)})

    print(stance)
    return render_template('bargraph.html', stance=stance)


@app.route("/team")
def team():
    return render_template('team.html')


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.run(debug=True)
