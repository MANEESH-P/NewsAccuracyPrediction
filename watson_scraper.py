import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions, KeywordsOptions
from watson_developer_cloud.natural_language_understanding_v1 import Features, MetadataOptions

from rake_nltk import Rake

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16', iam_apikey='4kxtefSt-VgDt3LGbteO7tv0eAczVWdvXJcMIKhHdJfo', url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')


response = natural_language_understanding.analyze(
    text='Ind vs Aus 1st T20, India vs Australia Highlights: Australia win by 3 wickets on last ball',
    features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=15))).get_result()

print(json.dumps(response, indent=2))


# response = natural_language_understanding.analyze(
#     url='https://indianexpress.com/article/sports/cricket/india-vs-australia-1st-t20-live-cricket-score-ind-vs-aus-live-streaming-vizag-5598755/',
#     features=Features(metadata=MetadataOptions())).get_result()

# print(json.dumps(response, indent=2))

# r = Rake()
# article_title = response["metadata"]["title"]

# r.extract_keywords_from_text(article_title)

# keywords = r.get_ranked_phrases()
# for i in range(len(keywords)):
#     new_keywords = new_keywords + keywords[i].split()
# print(keywords)

keywords = []
for keyword in response['keywords']:
    if keyword['relevance'] > 0.30 and len(keywords) < 8:
        keywords.append(keyword['text'])

print(keywords)
