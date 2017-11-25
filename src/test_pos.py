from nltk import word_tokenize, pos_tag
from nltk.tag import StanfordPOSTagger
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
input_text_raw=raw_input("enter the string: ")




words = input_text_raw.split() 


my_path = os.path.abspath(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(my_path, "../config/config.yml")
MAPPING_PATH = os.path.join(my_path, "../data/mapping.json")
TRAINDATA_PATH = os.path.join(my_path, "../data/traindata.txt")
LABEL_PATH = os.path.join(my_path, "../data/")

sys.path.insert(0, LABEL_PATH)
import trainlabel

with open(CONFIG_PATH,"r") as config_file:
	config = yaml.load(config_file)

os.environ['STANFORD_MODELS'] = config['tagger']['path_to_models']

exec_command = config['preferences']['execute']
 


tokens = nltk.word_tokenize(input_text_raw)
print(tokens)
st = StanfordPOSTagger(config['tagger']['model'],path_to_jar=config['tagger']['path'])
stanford_tag = st.tag(input_text_raw.split())
print("Tags")
print(stanford_tag)
