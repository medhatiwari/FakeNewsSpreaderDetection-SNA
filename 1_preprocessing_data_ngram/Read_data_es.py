# importing packages
import random
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

#Reading in Spanish meta data (spreader or not)

r = open('datasets/es/truth.txt', "r")
data = r.read().split("\n")
idk = [] #id
spreader = [] #yes or no

for line in data:
    l = line.split(":::")
    if len(l)>1:
        idk.append(l[0])
        spreader.append(l[1])

meta_data=pd.DataFrame()
meta_data["ID"]=idk
meta_data["spreader"]=spreader

#Reading in and concatenating Spanish tweets

pathlist = Path('datasets/es').glob('**/*.xml')
ids=[]
x_raw=[]
for path in pathlist:  ##iterate files
    head, tail = os.path.split(path)
    t=tail.split(".")
    author=t[0]
    ids.append(author)
    path_in_str = str(path)
    tree = ET.parse(path_in_str)
    root = tree.getroot()
    for child in root:
        xi=[]
        for ch in child:
            xi.append(ch.text)
        content = ' '.join(xi)
        x_raw.append(content)

text_data=pd.DataFrame()
text_data["ID"]=ids
text_data["Tweets"]=x_raw

#Merging meta data and text data to one dataframe


es_data = pd.merge(meta_data, text_data, how='inner', on = 'ID')