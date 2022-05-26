# Telegram Chat Bot Using Machine Learning
## Description: 
This is "just for fun" project based on SKlearn ML models. <Br>
The main idea is try to create a Russian language talkative bot which determines user's main intention based on the input phrase and give the most appropriate answer to him. <br>
A ready-made dictionary of intentions and examples of answers to them was used for bot training (JSON). <br>

## Models:
Three different models were tested to determine the user's intention in his phrase. Results below:
- Logistic Regression (model score = 0.3884)
- Random forest Classifier (model score = 0.8281) 
- MLP Classifier (model score = 0.8247) <br>
So, I decided to use MLP Classifier.

## Input Data Preparation filter_text():
The input word (phrase) is lowered to lower case then spaces and punctuation marks are removed from it. 

## Word Comparison is_match():
Two words are compared: the input and which the model predicts in the body of this function.

## Main Logic Func bot():
- 1) Filter input data;
- 2) Try to find the answer directly in the dictionary;
- 3) If not - use ML model intent predictiction and take the random answer example from this intent group;
- 4) Or use the Failure Phrases if the model score is not enough in any case.
- 5) Repit until input phrase == one of exit_phrases.

### Bot conversation example:
![image](https://user-images.githubusercontent.com/57821178/170299578-90d76e38-1a78-4fce-8a43-b945c403f658.png)
