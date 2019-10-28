Dataset: UCI_credit_card
Run command: python model2.py --mode{train,cross_val,predict} --input filename.csv
You can also pick up the dataset from the hive table, by uncommenting the "conn" and "read_sql"
The code trains the dataset on different models{Logistic Regression,Linear Discrimination analysis,K Neighbour,Decision Tree Classifier, Gaussian
, SVC,Random Forest. 
You can add few more models to train .
The code further picks up the best model and store it as a pkl file.
The cross_val and predict function can be accomplished without training the model again and again.
The ML-Flow is also implemented, so as to compare and store your previous runs along with the parameters.
