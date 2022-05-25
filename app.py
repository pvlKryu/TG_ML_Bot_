### The goal is to determine the user's intention by the input text and give the random answer from correct intent###

from distutils.log import Log
import json, re, nltk, random
from math import fabs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def main():
    # Open bot dictionary #
    config_file = open("big_bot_config.json", "r") 
    BIG_CONFIG = json.load(config_file)

    ### dict_keys(['intents', 'failure_phrases']) ###
    X = [] # Phrases
    y = [] # Intents

    for name, intent in BIG_CONFIG["intents"].items(): #one intent has few examples
        for example in intent["examples"]:
                X.append(example)
                y.append(name)
        for example in intent["responses"]:
                X.append(example)
                y.append(name)

    ### Preparing data for model training ###

    ## NLP Vectorization using SKlearn ##
    vectorizer = CountVectorizer()
    vectorizer.fit(X)

    ### ML ###
    # Text classification = class (intent) predictions by text (phrase) #

    ## 1) Log Reg using SKlearn - Not effective ##

    # model = LogisticRegression()
    # vecX = vectorizer.transform(X) 
    # model.fit(vecX,y)
    # print(model.score(vecX, y))

    # model score = 0.3884 :( bad

    ## 2) Random forest Classifier - Not bad ##
    # model = RandomForestClassifier()
    # vecX = vectorizer.transform(X) 
    # model.fit(vecX,y)
    # print(model.score(vecX, y))

    # model score = 0.8281 :) much better

    ## 3) MLP Classifier - best model
    model = MLPClassifier()
    vecX = vectorizer.transform(X) 
    model.fit(vecX,y)
    # print(model.score(vecX, y))
    # model score = 0.8247 :) same as Random forest Classifier 

    exit_phrases = ["Выйти", "Выключись", "Стоп", "Stop", "Finish", "Exit", "выйти", "выключись", "стоп", "stop", "finish", "exit"]
    print("Put your message: ")
    msg = ""
    while not msg in exit_phrases:
        msg = input()
        print("[BOT]: " + bot(msg, BIG_CONFIG, model, vectorizer))

### Main funcs ###

## Input text filter ##
def filter_text(text):
    text = text.lower()
    pattern = r'[^\w\s]'
    text = re.sub(pattern, "", text)
    return text

## Func return 1 if the texts match or 0 otherwise ##
def is_match(text1, text2):
    text1 = filter_text(text1)
    text2 = filter_text(text2)

    if len(text1) == 0 or len(text2) == 0:
        return False

    if text1.find(text2) != -1:
        return True

    # Levenshtein distance (edit distance = edit distance)
    distance = nltk.edit_distance(text1, text2)  # Number of characters [0...Inf]
    length = (len(text1) + len(text2))/2  # Average length of two texts
    score = distance / length  # 0...1

    return score < 0.6

## Get the intent by input text ##
def get_intent_ml(text, model, vectorizer):
    vec_text = vectorizer.transform([text])
    intent = model.predict(vec_text)[0]
    return intent

## Find the intent directly ##
def get_intent(text, BIG_CONFIG):
  for name, intent in BIG_CONFIG["intents"].items():
    for example in intent["examples"]:
      if is_match(text, example):
        #print(f"name={name} example={example}")
        return name

  return None


## Main bot logic func ##
def bot(phrase, BIG_CONFIG, model, vectorizer):

    # Filter input data #
    phrase = filter_text(phrase)

    # 1) Find the answer directly #
    intent = get_intent(phrase, BIG_CONFIG)

    if not intent:
    # 2) ML  #
        intent = get_intent_ml(phrase, model, vectorizer)

    # If intent found - choose random answer
    if intent:
        responses = BIG_CONFIG["intents"][intent]["responses"] 
        return random.choice(responses)  

    # 3) Failure Phrase #  
    failure = BIG_CONFIG["failure_phrases"]
    return random.choice(failure)

main()
