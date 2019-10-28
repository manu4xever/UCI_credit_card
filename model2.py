import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from pyhive import hive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from optparse import OptionParser
from sklearn.externals import joblib


def train(X, y):
	train.X_train, train.X_test, train.y_train, train.y_test = train_test_split(X, y, test_size=0.2)
	scX = StandardScaler()
	train.X_train = scX.fit_transform(train.X_train)
	train.X_test = scX.transform(train.X_test)
	classifier1 = SVC(gamma='auto', kernel="rbf" )
	classifier1.fit(train.X_train, train.y_train)
	j = joblib.dump(classifier1, 'SVM.pkl')
	
	with open ('obj.pkl','wb') as f:
		pickle.dump([train.X_train,train.X_test, train.y_train,train.y_test],f)
	
	
	
def cross_val():
	classifier1 = joblib.load('SVM.pkl')
	with open('obj.pkl','rb') as f:
		train.X_train,train.X_test, train.y_train,train.y_test=pickle.load(f)
	scoresSVC = cross_val_score(classifier1, train.X_train, train.y_train, cv=10)
	print("Mean kernel-SVM CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresSVC.mean(), scoresSVC.std()))

def predict():
	classifier1 = joblib.load('SVM.pkl')
	with open('obj.pkl','rb') as f:
		train.X_train,train.X_test, train.y_train,train.y_test=pickle.load(f)
		#'SVM.pkl'=pickle.load(f)
		#train.classifier1 = pickle.load('SVM.pkl')
	#classifier1 = SVC(gamma='auto', kernel="rbf", )
	y_pred = classifier1.predict(train.X_test)
	cm = confusion_matrix(train.y_test, y_pred)
	print("Accuracy on Test Set for kernel-SVM = %.2f" % ((cm[0, 0] + cm[1, 1]) / len(train.X_test)))

if __name__ == '__main__':
	print("Enter the command in the followning format")
	print("python filename.py --mode train/predict/ --input input_file");
	# connect odbc to data source name
	conn = hive.Connection(host="192.168.2.36", port=10000, username="cloudera" , database="creditcard")

	parser = OptionParser()
	parser.add_option("--mode", dest="mode", type="choice", choices=["train", "cross_val", "predict"], default="train")
	parser.add_option("--input", type="str")
	(options, args) = parser.parse_args()
	
	if options.mode in "train":
		assert options.input is not None
		#df = pd.read_csv(options.input, encoding='utf-8',nrows=500)
		#df=pd.read_csv("UCI_Credit_Card.csv",nrows=500)
		df = pd.read_sql("SELECT * FROM loaddataatrest", conn)
		#df = pd.read_sql("SELECT * FROM Uber_Book", conn)
		data = pd.DataFrame(df)
		
		#data.rename(str.upper, axis='columns')
		print((df.head()))
		
		data.columns={"ID", "LIMIT_BAL",  "SEX", "EDUCATION",  "MARRIAGE",
             "AGE",  "PAY_0",  "PAY_2",  "PAY_3",  "PAY_4",  "PAY_5",
              "PAY_6", "BILL_AMT1",  "BILL_AMT2",  "BILL_AMT3",
              "BILL_AMT4",  "BILL_AMT5",
              "BILL_AMT6",  "PAY_AMT1",  "PAY_AMT2",
             "PAY_AMT3", "PAY_AMT4","PAY_AMT5", "PAY_AMT6",
              "default_payment_next_month"}

		print(data.head())
		output = 'default_payment_next_month'
		data= data.dropna()

		cols = [f for f in data.columns if data.dtypes[f] != "object"]
		cols.remove("ID")
		cols.remove(output)

		

		
		# The quantitative vars:
		quant = ["LIMIT_BAL", "AGE"]

		# The qualitative but "Encoded" variables (ie most of them)
		qual_Enc = cols
		qual_Enc.remove("LIMIT_BAL")
		qual_Enc.remove("AGE")

		logged = []
		for ii in range(1, 7):
			qual_Enc.remove("PAY_AMT" + str(ii))
			data["log_PAY_AMT" + str(ii)] = data["PAY_AMT" + str(ii)].apply(lambda x: np.log1p(x) if (x > 0) else 0)
			logged.append("log_PAY_AMT" + str(ii))

		for ii in range(1, 7):
			qual_Enc.remove("BILL_AMT" + str(ii))
			data["log_BILL_AMT" + str(ii)] = data["BILL_AMT" + str(ii)].apply(lambda x: np.log1p(x) if (x > 0) else 0)
			logged.append("log_BILL_AMT" + str(ii))

		#f = pd.melt(data, id_vars=output, value_vars=logged)
		#g = sns.FacetGrid(f, hue=output, col="variable", col_wrap=3, sharex=False, sharey=False)
		#g = g.map(sns.distplot, "value", kde=True).add_legend()

		features = quant + qual_Enc + logged + [output]
		corr = data[features].corr()

		f, ax = plt.subplots(figsize=(10, 10))

		# Generate a custom diverging colormap
		cmap = sns.diverging_palette(220, 10, as_cmap=True)

		# Draw the heatmap with the mask and correct aspect ratio
		sns.heatmap(corr, cmap=cmap, vmin=0, vmax=1, center=0,
					square=True, linewidths=.5)

		# plt.subplots(figsize=(30,10))
		# sns.heatmap( corr, square=True, annot=True, fmt=".1f" )


		features = quant + qual_Enc + logged
		X = data[features].values
		y = data[output].values

		train(X, y)
		

		
	elif options.mode=="cross_val":
		assert options.input is not None
		cross_val()
	elif options.mode == "predict":
		predict()
	else:
		raise Exception("Invalid parser mode", options.mode)
  