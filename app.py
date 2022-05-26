### The goal is to determine the user's intention by the input text and give the random answer from correct intent###
from distutils.log import Log
import json, re, nltk, random
from math import fabs
from numpy import vectorize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters

def get_model(BIG_CONFIG):
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

    ## 2) Random forest Classifier - Best model ##
    model = RandomForestClassifier()
    vecX = vectorizer.transform(X) 
    model.fit(vecX,y)
    # print(model.score(vecX, y))
    # model score = 0.8281 :) Best result

        ## 3) MLP Classifier - not bad but too long and GPU expensive ##
        # model = MLPClassifier()
        # vecX = vectorizer.transform(X) 
        # model.fit(vecX,y)
        # print(model.score(vecX, y))
        # model score = 0.8247 :) same as Random forest Classifier 
    return model, vectorizer

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
def get_intent_ml(text):
    vec_text = vectorizer.transform([text])
    intent = model.predict(vec_text)[0]
    return intent

## Find the intent directly ##
def get_intent(text):
  for name, intent in BIG_CONFIG["intents"].items():
    for example in intent["examples"]:
      if is_match(text, example):
        #print(f"name={name} example={example}")
        return name

  return None


## Main bot logic func ##
def bot(phrase):

    # Filter input data #
    phrase = filter_text(phrase)

    # 1) Find the answer directly #
    intent = get_intent(phrase)

    if not intent:
    # 2) ML  #
        intent = get_intent_ml(phrase)

    # If intent found - choose random answer
    if intent:
        responses = BIG_CONFIG["intents"][intent]["responses"] 
        return random.choice(responses)  

    # 3) Failure Phrase #  
    failure = BIG_CONFIG["failure_phrases"]
    return random.choice(failure)

## TG Bot server logic func ##
def bot_telegram_reply(update: Update, ctx):
    
    text = update.message.text
    exit_phrases = ["Выйти", "Выключись", "Стоп", "Stop", "Finish", "Exit", "выйти", "выключись", "стоп", "stop", "finish", "exit"]

    if text in exit_phrases:
        update.message.reply_text("Bye-Bye")
        exit()

    reply = bot(text)
    update.message.reply_text(reply)
    name = update.message.chat.full_name
    print(f"[{name}] {text}: {reply}")



# Open bot dictionary #
config_file = open("big_bot_config.json", "r") 
BIG_CONFIG = json.load(config_file)

## Use ML func ##
model, vectorizer = get_model(BIG_CONFIG)

## Conect to TG server ##
f = open('BOT_KEY.txt')
BOT_KEY = f.read()
upd = Updater(BOT_KEY)

## Create MessageHandler ##
handler = MessageHandler(Filters.text, bot_telegram_reply)

## Register MessageHandler to Updater ##
upd.dispatcher.add_handler(handler)

print("It works")

## Start polling TG server ##

upd.start_polling()
upd.idle()

exit()