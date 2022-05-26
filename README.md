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
So, I decided to use Random forest Classifier because it is faster and less GPU expensive.

## Input Data Preparation filter_text():
The input word (phrase) is lowered to lower case then spaces and punctuation marks are removed from it. 

## Word Comparison is_match():
Two words are compared: the input and which the model predicts in the body of this function.

## Main Logic Func bot():
1. Filter input data;
2. Try to find the answer directly in the dictionary;
3. If not - use ML model intent predictiction and take the random answer example from this intent group;
4. Or use the Failure Phrases if the model score is not enough in any case.
5. Repit until input phrase == one of exit_phrases.

### Bot conversation example:
## Server output:
![image](https://user-images.githubusercontent.com/57821178/170508597-dfba3ccd-40d1-4268-81c1-c7f32be1b43a.png)
As you can see, users names are shown.
## Telegram Output:
![image](https://user-images.githubusercontent.com/57821178/170508918-813fc066-a24c-4f5a-b348-be7442faab0f.png)
