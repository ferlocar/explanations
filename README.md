# explanations

Explanation for category variables updated (in 'explanations/explanation/explanations')

Under 'explanations/explanation/explanations', run
~~~~ 
python tester_new.py --model MODEL_NAME
~~~~
will produce 'explanations/explanation/files/explanations_MODEL_NAME.csv'.
Or run
~~~~ 
python tester_new.py --model MODEL_NAME --exp
~~~~
will produce 'explanations/explanation/files/explanations_MODEL_NAME_exp.csv'

MODEL_NAME allow 'lc' (LogisticRegression) and 'rf' (RandomForest) for now.

LendClub Data also added

If you want to understand the meaning of the feature name, please go to the notebook
