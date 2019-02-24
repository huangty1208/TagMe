### Copy right Ty Huang
### None of this will work if you just cp my code
### models were deposited in another repositroy
#!flask/bin/python

import re
import pickle
import numpy as np
import pandas as pd
from gensim.models import FastText

with open('stop.txt') as f:
    lines = f.read().splitlines()
    
stop_words = sorted(set(lines))
f.close()



filename = 'tfidf_model1n.sav'
X = pickle.load(open(filename, 'rb'))
filename = 'tfidf_model1ncv.sav'
cv = pickle.load(open(filename, 'rb'))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.wordnet import WordNetLemmatizer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names=cv.get_feature_names()


from scipy.spatial import distance

tag_v_2d = np.load('tag_v_2d.npy')
tagname_2d = np.load('tagname_2d.npy')
tagcounts_2d = np.load('tagcounts_2d.npy')



#Function for sorting tf_idf in descending order
from scipy.sparse import coo_matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



model2 = FastText.load("C:\\Users\\huang\\Google Drive\\Insight\\Extension\\tagme\\fasttext.model.1.6m")



from flask import Flask, render_template, request, redirect, Response, jsonify


app = Flask(__name__)



@app.route('/')
def worker():
    return "main"


@app.route('/_process', methods=["GET", "POST"])
def calculate():

    content = request.json

    print(content)
 
    doc = content["content"]

    doc = doc.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:—_-,./<>?\|`~-=+"})
    doc = doc.translate ({ord(c): "" for c in "'"})
    doc = doc.lower()
    doc = doc.split()
    tem = []
    lem = WordNetLemmatizer()
    for word in doc: 
        text = word.strip()    
        text=re.sub("(\\d|\\W)+"," ",text)  

        text = text.replace(" ", "")

    
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        # remove special characters and digits
      

        tem.append(text)

    text = [lem.lemmatize(t) for t in tem if not t in stop_words]    
    text = " ".join(text)

    #sort the tf-idf vectors by descending order of scores
    tf_idf_vector=tfidf_transformer.transform(cv.transform([text]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,5)


    key_three = [] 
    for k in keywords:
        key_three.append(k)
    

    print(key_three[:3])
	
	
    collect_name = []
    collect_count = []

    test_list = key_three[:3]


    for i in test_list:
    
        test_v = model2.wv[i]

        distances = distance.cdist([test_v], tag_v_2d, "cosine")[0]

        ind = np.argpartition(distances, 5)[:5]

        min_distance = distances[ind]

        min_name = tagname_2d[ind]
    
        min_count = tagcounts_2d[ind]
    

        collect_name.extend(min_name)
        collect_count.extend(min_count)


    dfc = pd.DataFrame({
    'name' : collect_name,
    'count' : collect_count
    })

    dfc = dfc.drop_duplicates()
    dfc_sorted =  dfc.sort_values(['count'] ,ascending=False)

    dfToList = dfc_sorted['name'].tolist()[:9]
    countToList = dfc_sorted['count'].tolist()[:9]

    y = ""
    for i in dfToList:
        y += "#"+i+"\n"

    return jsonify({"tag": y})
	
	

if __name__ == '__main__':
	# run!
    app.debug = True
    app.run()

