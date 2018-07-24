import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
import math
import tkinter
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers.core import Dense , Dropout , Activation , Flatten
from keras.layers import merge
from scipy import spatial
from keras import backend as K
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from keras.utils.np_utils import to_categorical
from tkinter import *
from tkinter import Entry, Button, mainloop, Frame, StringVar, BOTTOM,S,messagebox,Text, Tk

# size of the word embeddings
embeddings_dim = 200

# maximum number of words to consider in the representations
max_features = 300000

# maximum length of a sentence
max_sent_len = 500

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes1 = 5
num_classes2 = 3
fields = 'T.C', 'İsim', 'Soyisim', 'Öğrenci Numarası'
question1 = "1-) Midesi alınan bir bireyin yaşayacağı sorunlar nelerdir? (5 puan)"
question2 = "2-) Atom ve molekül arasındaki farklar nelerdir? (3 puan)"
answer1_ = "midesi olmadan proteini sindiremez aç kalır"
answer2_ = "atom maddenin yapı taşıdır molekül ise atomlardan oluşur"

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
   
def findGradeFromLabel(label):
    i = 0
    for l in label:
        if l == 1:
            return i;
        i += 1

def maxTermCountInDocument(oneGradeManyAnswer):
	res = 0
	for answer in oneGradeManyAnswer:
		val = int(oneGradeManyAnswer[answer])
		if val > res:
			res = oneGradeManyAnswer[answer]
	return res
def dfi(allGradeManyAnswer, word):
	res = 0
	i = 0
	while i < len(allGradeManyAnswer):
		for answer in allGradeManyAnswer[i]:
			if word in answer:
				res += 1
		i += 1
	return res
def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)   
    return z
def generateOccurrences(train_labels, train_texts):
	allQuestions = {}   
	for x in range(0,4):
	    question = dict()
	    i = 0
	    while i < len(train_labels):
	        #print("label :" + repr(train_labels[i]) + " x :" + repr(x))
	        if int(train_labels[i]) == x:
	           # print("içerde")
	            for j in train_texts[i].split():
	               # print("baya içerde")
	                if j in question:
	                    question[j] += 1
	                else:
	                    question[j] = 1
	        allQuestions[x] = question
	        i += 1
	return allQuestions
def tfIdf(grade, allQuestions, word, answerCount):
	tfIdfValue = 0
	maxF = maxTermCountInDocument(allQuestions[grade])

	if word in allQuestions[grade]:
		tfValue = int(allQuestions[grade][word]) / maxF;
	else:
		tfValue = 1 / maxF;		#smoothing

	dfiValueDown = dfi(allQuestions,word)
	if dfiValueDown == 0:
		tfIdfValue = 0
	else:
		dfiValue = answerCount / dfiValueDown;
		tfIdfValue = tfValue*math.log2(dfiValue);
	return tfIdfValue
def knn(lstmGrade, knnValue, similarities, labels, class_num):
	answers = []
	j = 0
	while j < class_num:
		a = (j, 0)
		answers.append(a)
		j += 1
	i = 0
	while i < knnValue:
		j = 0
		while j < class_num:
			if answers[j][0] == findGradeFromLabel(labels[similarities[i][0]]):
				answers[j] = list(answers[j])
				answers[j][1] = answers[j][1] + 1
			j += 1
		i += 1
	maxSimNeighbour = 0
	maxSimNeighbourIndex = -1
	i = 0
	for ans in answers:
		if ans[1] > maxSimNeighbour:
			maxSimNeighbour = ans[1]
			maxSimNeighbourIndex = i
		i += 1
	i = 0
	avg = 0
	counter = 0 
	while i < knnValue:
		grade_temp = findGradeFromLabel(labels[similarities[i][0]])
		if grade_temp == maxSimNeighbourIndex:
			avg += similarities[i][1]
			counter += 1
		i += 1
	if maxSimNeighbour == 0:
	    avg = avg / 1
	else:
	    avg = avg / counter
	if avg > 0.75 and maxSimNeighbourIndex != 0 and answers[lstmGrade][1] == 0:
	    return maxSimNeighbourIndex
	"""if lstmGrade == 0 and answers[lstmGrade][1] == 0:
		return maxSimNeighbourIndex"""
	return lstmGrade

def calculator(model, train_labels, train_texts, tokenizer, question, answer, class_num):
	print(question, "\n")
	print("Aswer :", answer, "\n")
	liste = []
	str1 = answer
	#print("You entered:", str1)
	liste.append(str1.lower())
	#print(liste)
	train_sequencesss = sequence.pad_sequences( tokenizer.texts_to_sequences( liste ) , maxlen=max_sent_len )
	results = model.predict_classes( train_sequencesss )
	print("\nYour answer's grade (lstm) is " + repr(results[0]))
	i = 0
	start_sim = -1
	start_index = 0
	s1_afv = avg_feature_vector(str1.lower(), model=embeddings, num_features=200, index2word_set=index2word_set)
	answerAndSimilarity = []
	while i < len(train_texts):
	    s2_afv = avg_feature_vector(train_texts[i], model=embeddings, num_features=200, index2word_set=index2word_set)
	    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
	    ansSim = (i, sim)
	    answerAndSimilarity.append(ansSim)
	    i += 1
	#print("\nstart index :" + repr(start_index))
	b = sorted(sorted(answerAndSimilarity, key = lambda x : x[0]), key = lambda x : x[1], reverse = True)
	i = 0
	#print("\n")
	while i < 3:
	    print("-------------------" + repr(i+1) + "--------------------")
	    #print("answer :" + repr(train_texts[b[i][0]]))
	    print("grade :" + repr(findGradeFromLabel(train_labels[b[i][0]])))
	    print("similarity :" + repr(b[i][1]))
	    i += 1
	last_grade = knn(results[0], 3, b, train_labels, class_num)
	print("\nLast Grade :" + repr(knn(results[0], 3, b, train_labels, class_num)))
	print("_________________________________________________")
	print("\n------------------------------------------------------------\n")
	return last_grade

def calculate(entries):
	answer1_ = answer1.get("1.0","end-1c")
	answer2_ = answer2.get("1.0","end-1c")

	if answer1_ == "" or answer2_ == "":
   	  	var = messagebox.showinfo("Uyarı" , "Eksik bilgi girdiniz!")
   	  	return False
	first_result_int = calculator(model1, train_labels1, train_texts1, tokenizer1, question1, answer1_, 7)
	second_result_int = calculator(model2, train_labels2, train_texts2, tokenizer2, question2, answer2_, 5)

	first_result_str = "1.soru  =>  " + str(first_result_int) + "\n"
	second_result_str = "2.soru  =>  " + str(second_result_int) + "\n"

	all_result_int = first_result_int + second_result_int
	all_result_str = first_result_str + second_result_str
	all_result_str += "\nToplam puan : " + str(all_result_int)

	all_result.config(text=all_result_str)

	output = open(file_name,'a',encoding="utf8")
	output.write("\n1-) => " + repr(answer1_) + "\n")
	output.write("puan :" + repr(first_result_int) + "\n")
	output.write("2-) => " + repr(answer2_) + "\n")
	output.write("puan :" + repr((second_result_int)) + "\n")
	output.write("\nSınav Puanı :" + repr((all_result_int))+ "\n")
	output.write("_________________________________________________________________________\n")
	output.close()

def iptal():
	K.clear_session()
	sys.exit()
def fetch(entries):
   i = 0
   for entr in entries:
   	  txtt = entr[1].get()
   	  if txtt == "":
   	  	var = messagebox.showinfo("Uyarı" , "Eksik bilgi girdiniz!")
   	  	return False
   output = open(file_name,'a',encoding="utf8")
   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print('%s: "%s"' % (field, text))
      if i == 0:
      	output.write("T.C :" + repr(text) + "\n")
      elif i == 1:
      	output.write("Name :" + repr(text) + "\n")
      elif i == 2:
      	output.write("Surname :" + repr(text) + "\n")
      elif i == 3:
      	output.write("Öğrenci Numarası :" + repr(text) + "\n\n")
      i += 1
   output.close()
   root.destroy() 

def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=15, text=field, anchor='w')
      ent = Entry(row)
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   return entries


 
if len(sys.argv) < 3:
	print("\nWrong command line argument!!!\n")
	print("python3 Demo.py <File_Name_For_Saving_Results> <Student_Count>\n")
	sys.exit()

file_name = str(sys.argv[1])
student_count = int(sys.argv[2])
if student_count == 0:
	print("\nWrong student count argument!!!\n")
	print("python3 Demo.py <File_Name_For_Saving_Results> <Student_Count>\n")
	sys.exit()
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
#embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 


#embeddings = gensim.models.KeyedVectors.load_word2vec_format('AllDataWithLastUpdatedVectors.txt', binary=False, unicode_errors='ignore')
embeddings = np.load('my_vector_dictionary.npy').item()
index2word_set = set(embeddings.index2word)

data1 = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("answer270WithRandomShuffle.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data )

data11 = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("dataAllTester.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data1 )

print(int(len(data1)))
train_size1 = int(len(data1))
test_size1 = int(len(data11))
train_texts1 = [ txt.lower() for ( txt, label ) in data1[0:train_size1] ]
test_texts1 = [ txt.lower() for ( txt, label ) in data11[0:test_size1] ]

train_labels1 = [ label for ( txt , label ) in data1[0:train_size1] ]
test_labels1 = [ label for ( txt , label ) in data11[0:test_size1] ]

num_classes1 = len( set( train_labels1 + test_labels1 ) )
print("Number of Classes :" + repr(num_classes1))
print("Size of Train :" + repr(train_size1))

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(train_texts1)

train_sequences1 = sequence.pad_sequences( tokenizer1.texts_to_sequences( train_texts1 ) , maxlen=max_sent_len )
test_sequences1 = sequence.pad_sequences( tokenizer1.texts_to_sequences( test_texts1 ) , maxlen=max_sent_len )
train_matrix1= tokenizer1.texts_to_matrix( train_texts1 )
test_matrix1 = tokenizer1.texts_to_matrix( test_texts1 )
embedding_weights1 = np.zeros( ( max_features , embeddings_dim ) )

allQuestions1 = generateOccurrences(train_labels1, train_texts1)

for word,index in tokenizer1.word_index.items():
  if index < max_features:
    try: embedding_weights1[index,:] = embeddings[word]
    except: embedding_weights1[index,:] = np.random.rand( 1 , embeddings_dim )

print("-------------------------- First Question-Answer ------------------------------\n")
#le.fit( train_labels + test_labels )
train_labels1 = to_categorical( train_labels1 )
test_labels1 = to_categorical( test_labels1 )



np.random.seed(0)
model1 = Sequential()
model1.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights1] ))
model1.add(Dropout(0.25))
model1.add(LSTM(activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid', return_sequences=True))
model1.add(Dropout(0.25))
model1.add(LSTM( activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid'))
model1.add(Dropout(0.25))
model1.add(Dense(6))
model1.add(Activation('sigmoid'))
model1.load_weights('q270DataBatch16Epoch40Shuffled.h5')

#---------------------------------------------------2.soru ----------------------------------------------

data2 = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("answer170WithRandomShuffle3.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data )
data22 = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("dataAllTester3.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
#random.shuffle( data1 )

print(int(len(data2)))
train_size2 = int(len(data2))
test_size2 = int(len(data22))
train_texts2 = [ txt.lower() for ( txt, label ) in data2[0:train_size2] ]
test_texts2 = [ txt.lower() for ( txt, label ) in data22[0:test_size2] ]

train_labels2 = [ label for ( txt , label ) in data2[0:train_size2] ]
test_labels2 = [ label for ( txt , label ) in data22[0:test_size2] ]

num_classes2 = len( set( train_labels2 + test_labels2 ) )
print("Number of Classes :" + repr(num_classes2))
print("Size of Train :" + repr(train_size2))

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(train_texts2)

train_sequences2 = sequence.pad_sequences( tokenizer2.texts_to_sequences( train_texts2 ) , maxlen=max_sent_len )
test_sequences2 = sequence.pad_sequences( tokenizer2.texts_to_sequences( test_texts2 ) , maxlen=max_sent_len )
train_matrix2 = tokenizer2.texts_to_matrix( train_texts2 )
test_matrix2 = tokenizer2.texts_to_matrix( test_texts2 )
embedding_weights2 = np.zeros( ( max_features , embeddings_dim ) )

allQuestions2 = generateOccurrences(train_labels2, train_texts2)

for word,index in tokenizer2.word_index.items():
  if index < max_features:
    try: embedding_weights2[index,:] = embeddings[word]
    except: embedding_weights2[index,:] = np.random.rand( 1 , embeddings_dim )

print("-------------------------- Second Question-Answer -----------------------------\n")
#le.fit( train_labels + test_labels )
train_labels2 = to_categorical( train_labels2 )
test_labels2 = to_categorical( test_labels2 )



np.random.seed(0)
model2 = Sequential()
model2.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights2] ))
model2.add(Dropout(0.25))
model2.add(LSTM(activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid', return_sequences=True))
model2.add(Dropout(0.25))
model2.add(LSTM( activation='sigmoid', units = 200, recurrent_activation='hard_sigmoid'))
model2.add(Dropout(0.25))
model2.add(Dense(4))
model2.add(Activation('sigmoid'))
model2.load_weights('q170DataBatch16Epoch40Shuffledq3.h5')


output = open(file_name,'a',encoding="utf8")
output.write("------------------------------- Questions ---------------------------\n\n")
output.write(question1 + "\n\n")
output.write(question2 + "\n\n")
output.write("---------------------------------------------------------------------\n\n")
output.write("_______________________________ Answers _____________________________\n\n")
output.close()
i = 0
while i < student_count:
	root = tkinter.Tk()
	root.title("Kişisel Bilgiler")
	ents = makeform(root, fields)
	root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
	b1 = Button(root, text='Kaydet',
	      command=(lambda e=ents: fetch(e)))
	b1.pack(side=LEFT, padx=5, pady=5)
	b2 = Button(root, text='İptal', command=iptal)
	b2.pack(side=LEFT, padx=5, pady=5)
	root.mainloop()



	pencere = tkinter.Tk()

	pencere.title("Açık Uçlu Sınav Sorularının Otomatik Olarak Değerlendirilmesi")
	pencere.geometry("700x650")
	 
	uygulama = Frame(pencere)
	uygulama.grid()
	 
	L1 = Label(uygulama, text="\n\n1-) Midesi alınan erişkin bir bireyin yaşayacağı sorunlar nelerdir? (5 puan)")
	L1.grid(padx=120, pady=10)
	 
	answer1 = Text(uygulama, height=10, width=40)
	answer1.grid(padx=30, pady=10)
	   
	L2 = Label(uygulama, text="2-) Atom ve molekül arasındaki fark nedir? (3 puan)")
	L2.grid(padx=110, pady=10)
	 
	answer2 = Text(uygulama, height=10, width=40)
	answer2.grid(padx=30, pady=10)


	uygulama.bind('<Return>', (lambda event, e=ents: calculate(e))) 
	button1 = Button(uygulama, text = "Gönder" , width=20, command=(lambda e=ents: calculate(e)))
	button1.grid(padx=110, pady=20)

	all_result = Label(uygulama, text="")
	all_result.grid(padx=110, pady=10)
	
	pencere.mainloop()

	i += 1

K.clear_session()