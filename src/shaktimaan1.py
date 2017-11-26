import yaml  
import sys
import random
import nltk
import operator
import jellyfish as jf
import json
import requests
import os
import time
import signal
import subprocess
from nltk.tag import StanfordPOSTagger
from textblob.classifiers import NaiveBayesClassifier
#sfrom execute import construct_command
#from feedback import get_user_feedback
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

#nltk.download('punkt')
def signal_handler(signal, frame):
    print ('Thank You!')
    sys.exit(0)



data={
	'change': {
		'chgrp':["group"],
		'chmod':["permission"],
		'chown':["ownership"],
		'passwd':["password"],
	},
	'display': {
		'ls':["contents","list","files","current", "directory"],
		'cat':["concatenate","display", "combine", "print"],
		'dirname':["directory","name",],
		'echo':["text"],
		'less':["less"],
		'more':["more"],
		'head':["first","lines"],
		'tail':["last","lines"],
		'man':["manual","help"],
		'ps':["process","active","running"],
		'who':["logged","user", "username"],
		'whoami':["current","user", "username"],
		'cal':["calender"],
		'date':["date"],
		'pwd':["working","directory"],
	},
	'create': {
		'mkdir':[ "directory", "folder"],
		'mkfifo':["named", "pipe"],
		'mknod':["special","file",],
		'touch':["file","update", "timestamp"],
		'ln':["symbolic", "hard", "link"],
		
	},
	'compare': {
		'cmp':[ "binary", "files"],
		'diff':["text", "files"],
		
	},
	'search': {
		'grep':[ "match", "regular", "expression"],
		
	},
}

command_format={
	"ls":[],
	"cat":["noun", "noun"],
	"dirname":["string"],
	"echo":["string"],
	"more":["noun"],
	"less":["noun"],
	"head":["noun"],
	"tail":["noun"],
	"man":["noun"],
	"ps":[],
	"who":[],
	"whoami":[],
	"cal":[],
	"date":[],
	"pwd":[],
	"mkdir":["noun"],
	"mkfifo":["noun"],
	"mknod":["noun"],
	"touch":["noun"],
	"ln":["noun", "noun"],
	"cmp":["noun", "noun"],
	"diff":["noun", "noun"],
	"grep":["noun", "noun"],
	"chgrp":[],
	"chmod":[],
	"chown":[],
	"passwd":[],
}


noun=[]
adjective=[]
verb=[]
adverb=[]
determiner=[]
pronoun=[]
modal=[]
particle=[]
symbol=[]
cardinal=[]
conjuction=[]
preposition=[]
interjection=[]
existential=[]


signal.signal(signal.SIGINT, signal_handler)

my_path = os.path.abspath(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(my_path, "../config/config.yml")
MAPPING_PATH = os.path.join(my_path, "../data/mapping1.json")
TRAINDATA_PATH = os.path.join(my_path, "../data/traindata1.txt")
LABEL_PATH = os.path.join(my_path, "../data/")

sys.path.insert(0, LABEL_PATH)
import trainlabel1

with open(CONFIG_PATH,"r") as config_file:
    config = yaml.load(config_file)

os.environ['STANFORD_MODELS'] = config['tagger']['path_to_models']

exec_command = config['preferences']['execute']

    
def classify(text):
    X_train = np.array([line.rstrip('\n') for line in open(TRAINDATA_PATH)])
    y_train_text = trainlabel1.y_train_text
    #print y_train_text
    #print X_train
    X_test = np.array([text])
    #target_names = ['file', 'folder', 'network', 'system', 'general']

    lb = preprocessing.MultiLabelBinarizer()
    Y = lb.fit_transform(y_train_text)
    #print Y
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])
   # classifier=OneVsRestClassifier(LinearSVC())
    classifier.fit(X_train, Y)
    predicted = classifier.predict(X_test)
    all_labels = lb.inverse_transform(predicted)

    for item, labels in zip(X_test, all_labels):
        return (', '.join(labels))

def suggestions(suggest_list):
    suggest = (sorted(suggest_list,reverse=True)[:5])
    return suggest

def execute_command(command):
	import subprocess
	p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
	output, err = p.communicate()
	print  output

def find_arg(category,stanford_tag):
	for item in stanford_tag:
		if item[1]=="VB" or item[1]=="VBD" or item[1]=="VBG" or item[1]=="VBN" or item[1]=="VBP" or item[1]=="VBZ":
	 		verb.append(item[0])
		elif item[1]=="NN" or item[1]=="NNS" or item[1]=="NNP" or item[1]=="NNPS":
			noun.append(item[0])
		elif item[1]=="JJ" or item[1]=="JJR" or item[1]=="JJS":
			adjective.append(item[0])
		elif item[1]=="RB" or item[1]=="RBR" or item[1]=="RBS":
			adverb.append(item[0])
		elif item[1]=="WDT" or item[1]=="PDT" or item[1]=="DT":
			determiner.append(item[0])
		elif item[1]=="PRP" or item[1]=="PRP$" or item[1]=="WP$":
	    		pronoun.append(item[0])
	        elif item[1]=="MD":
	    		modal.append(item[0])
	        elif item[1]=="RP":
	    		particle.appen(item[0])
       	        elif item[1]=="SYM":
	    		symbol.append(item[0])
	        elif item[1]=="CD":
	    		cardinal.append(item[0])
	        elif item[1]=="CC":
			conjunction.append(item[0])
		elif item[1]=="IN":
			preposition.append(item[0])
		elif item[1]=="UH":
			interjection.append(item[0])
		elif item[1]=="EX":
			existential.append(item[0])

		


	print"verbs are"
	for item in verb:
		print item
	print "nouns are"
	for item in noun:
		print item

def construct_command(category):
	myformat=command_format[category]
	mycommand=""
	mycommand= mycommand+ str(category)+ str(" ")
	for item in myformat:
		if item=="noun":
			mycommand=mycommand+str(noun[0])+str(" ")
			del noun[0]
		elif item=="verb":
			mycommand=mycommand+str(verb[0])+str(" ")
			del noun[0]
	
	print mycommand
	print "executing..."
	execute_command(mycommand)




def call_reia():
    		max_score = 0.1
		
		
		print('-----------------------')
		user_input = raw_input("enter the string: ")
		#user_name = get_username(first_line.split(' ', 1)[0])
		suggest_list = []
		suggest_message = ""
		#prev_ts = ts
		print("\nINPUT = ")
		print(user_input)
		label = classify(user_input)
		if label == "":
			post_message("Sorry, I could not understand. Please rephrase and try again.")
			consume_message()
			
		print("Classified as : "+str(label))
		
		cnt=0;
		
			
		#print(stanford_tag)
		tokens = nltk.word_tokenize(user_input)
		print(tokens)
		sentence_tokens= []
		for i in tokens:
			if i == label:
				continue
			sentence_tokens.append(i)
		#print "sent token"
		#for item in sentence_tokens:
		#	print item

		#with open(MAPPING_PATH,'r') as data_file:    
		#	data = json.load(data_file)
		maxlabel=0
		#maxcnt=0
		category=""
		"""		
		for comm in data[label]:
			print comm
			for item in data[label][comm]:
				print item
		"""
		for comm in data[label]:
			cnt=0
			#maxcnt=0

			for item in data[label][comm]:
				for i in sentence_tokens:
					dist = jf.jaro_distance(unicode((item),encoding="utf-8"), unicode(str(i),encoding="utf-8"))
					if dist>=0.8:
						cnt=cnt+1
						#print "i "+i+"item "+item

			if cnt>maxlabel:
				print "comm is:"+comm
				maxlabel=cnt
				category=comm
		print "category is:"+category

		mylist= data[label][category]
		new_tokens= []
		
		for item in sentence_tokens:
			flag=0
			for i in mylist:
				dist= jf.jaro_distance(unicode((item),encoding="utf-8"), unicode(str(i),encoding="utf-8"))
				if dist>=0.8:
					flag=1
			if(flag==0):
				new_tokens.append(item)

		print("printing new_tokens")
		print new_tokens

		str1 = ' '.join(new_tokens)

		print str1

		st = StanfordPOSTagger(config['tagger']['model'],path_to_jar=config['tagger']['path'])
		stanford_tag = st.tag(str1.split())

		print("Tags")
		print stanford_tag

    	
    
    
		print "finding the arguments of the command "
		find_arg(category,stanford_tag)
		
		construct_command(category)
		



	
call_reia()









