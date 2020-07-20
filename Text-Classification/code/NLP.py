import tushare as ts
import pandas as pd
import csv
import re
import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import gensim


N = 80


def filter_special(desstr,restr=''):
    res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)

def split_word(line, stopword):
	split_list = jieba.cut(line,cut_all = False)

	split_sentence = ""
	for word in split_list:
		if (word not in stopword) and (word != '\t'):
			split_sentence += word + " "

	keyword_list = jieba.analyse.textrank(split_sentence)
	
	for word in keyword_list:
		split_sentence += word + " "
	return split_sentence.strip()
	
# get data and merge
def get_merge_data():
    pro = ts.pro_api('a6dae538a760f0b9e39432c1bff5e50a1c462a1a087e994dae18fa04')
    df0 = pro.stock_company(exchange='SZSE', fields='ts_code,business_scope')
    df1 = df0.dropna(axis = 0, how = 'any')
    df2 = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
    df1.to_csv("./data/corpus.csv")
    df2.to_csv("./data/label.csv")

    merged_df = pd.merge(df1,df2,how = 'right')
    # filter by number of records
    nonan_df = merged_df.dropna(axis=0, how='any')
    vc  = nonan_df['industry'].value_counts()
    pat = r'|'.join(vc[vc>N].index)          
    res_df  = nonan_df[nonan_df['industry'].str.contains(pat)]
	
    res_df.to_csv("./data/res.csv")

def change_label2num():
	num_label = set()
	num_labels = {}
	index = []
	stock_id = []
	content = []
	type_name = []
	with open("./data/res.csv") as res:
		reader = csv.reader(res)
		cnt = 0
		for line in reader:
			if (cnt == 0):
				cnt+=1
				continue
			index.append(line[0])
			stock_id.append(line[1])
			content.append(line[2])
			type_name.append(line[3])
			num_label.add(line[3])
		cnt = 0
		for type_content in num_label:
			num_labels[type_content] = cnt
			cnt += 1

	with  open("./data/res_label_num.csv","w",newline="",encoding="utf-8") as f:
		writer = csv.writer(f)
		for i in range(len(index)):
			writer.writerow([index[i],stock_id[i],content[i],num_labels[type_name[i]]])
	return num_labels
				


def TF_IDF():
	cleaned_data = []
	with open("./stopWord.txt") as stop:
		stopwords = set()
		for line in stop:
			stopwords.add(line.strip())
	with open("./data/res_label_num.csv") as csv_file:
		reader = csv.reader(csv_file)
		for line in reader:
			no_special_line = filter_special(line[2])
			split_no_stop_line = split_word(no_special_line,stopwords)
			cleaned_data.append(split_no_stop_line)

	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(cleaned_data))
	weight = tfidf.toarray()
	np.savetxt("./data/TF_IDF.csv",weight,fmt="%s",delimiter=",")


def classify_tfidf():
    y = []
    with open("./data/res_label_num.csv","r",encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            y.append(line[3])

    X = np.loadtxt("./data/TF_IDF.csv",delimiter=",")
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    result = metrics.classification_report(y_test,y_predict)
    print(result)

def classify_word2vec():
	# load model
	model = gensim.models.KeyedVectors.load_word2vec_format("./wiki.zh.text.vector", binary=False)


	cleaned_data = []
	labels = []
	with open("./stopWord.txt") as stop:
		stopwords = set()
		for line in stop:
			stopwords.add(line.strip())
	with open('./data/res_label_num.csv') as f:
		reader = csv.reader(f)
		for line in reader:
			labels.append(line[3])
			no_special_line = filter_special(line[2])
			split_no_stop_line = split_word(no_special_line,stopwords)
			cleaned_data.append(split_no_stop_line)
		cnt = 0
		for sentence in cleaned_data:
			words = sentence.split(" ")
			vector = []
			for word in words:
				word = word.replace('\n','')
				try:
					vector.append(model[word])
				except:
					print('Cannot find word in word2vec corpus')
					continue
			sentences_vectors.append(sum(np.array(vector)) / len(vector))
			sentences_y.append(labels[cnt])
			cnt += 1
		X = np.array(sentences_vectors)
		y = np.array(sentences_y)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
		model = MultinomialNB()
		model.fit(X_train, y_train)
		y_predict = model.predict(X_test)
		result = metrics.classification_report(y_test,y_predict)
		print(result)

if __name__ == '__main__':
	get_merge_data()
	change_label2num()
	TF_IDF()
	classify_tfidf()
	#classify_word2vec()

