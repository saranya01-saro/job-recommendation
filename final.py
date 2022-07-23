# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:50:45 2021

@author: regun
"""
import pandas as pd
import re
from gensim.utils import simple_preprocess
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def remove_punctuation(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()

def remove_stopwords(text,STOPWORDS):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def lemmatize_words(text,lemmatizer,wordnet_map):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text if len(word) >2])

def remove_escape(text):
    new_str = unicodedata.normalize("NFKD",text )
    return new_str

def preprocessor(text):
    text = text.replace('\\r', '').replace('&nbsp', '').replace('\n', '')
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

def minDistance(dist, sptSet):
 
    # Initialize minimum distance for next node
    min = 1000
 
    # Search not nearest vertex not in the
    # shortest path tree
    for u in range(len(dist)):
        if dist[u] < min and sptSet[u] == False:
            min = dist[u]
            min_index = u
 
    return min_index

def printSolution(dist):
    print("Vertex \tDistance from Source")
    for node in range(len(dist)):
        print(node, "\t", dist[node])
 
def GBA(distance,src):
    dist = list(distance[src])
    dist[src] = 0
    sptSet = [False] * distance.shape[0]
     
    for cout in range(distance.shape[0]):
     
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # x is always equal to src in first iteration
        x = minDistance(dist, sptSet)
     
        # Put the minimum distance vertex in the
        # shortest path tree
        sptSet[x] = True
     
        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for y in range(distance.shape[0]):
            if distance[x][y] > 0 and sptSet[y] == False and dist[y] > dist[x] + distance[x][y]:
                    dist[y] = dist[x] + distance[x][y]
    s = np.array(dist)
    sort_index = np.argsort(s)
    predicted = sort_index[:20]
    return predicted


jobs = pd.read_excel("internshalla_extracted1.xlsx")
jobs = jobs.drop(['Unnamed: 0'],axis=1)
jobs = jobs.dropna()
jobs.rename(columns={'Job_title':'Title','Job_Details':'Description','Job_Skills':'Skills','Job_location':'Location'},inplace=True)
#jobs["Description"] = jobs["Title"] + " " + jobs["Description"]


jobs["Description"] = jobs["Description"].apply(lambda text: text.replace("Key responsibilities",""))
jobs.Description = jobs.Description.str.replace('\d+', '')
STOPWORDS = stopwords.words('english')
jobs["Description"]= pd.DataFrame(jobs["Description"].str.lower())
jobs["Description"] = jobs["Description"].apply(lambda text: remove_stopwords(text,STOPWORDS))
jobs["Description"] = jobs["Description"].apply(lambda text: remove_punctuation(text))
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "J":wordnet.ADJ, "R":wordnet.ADV}
jobs["Description"] = jobs["Description"].apply(lambda text: lemmatize_words(text,lemmatizer,wordnet_map))
jobs["Description"] = jobs["Description"].apply(lambda text: remove_escape(text))
jobs["Description"]=jobs.Description.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in STOPWORDS) )

jobs.Skills = jobs.Skills.str.replace('\d+', '')
STOPWORDS = stopwords.words('english')
jobs["Skills"]= pd.DataFrame(jobs["Skills"].str.lower())
jobs["Skills"] = jobs["Skills"].apply(lambda text: remove_stopwords(text,STOPWORDS))
jobs["Skills"] = jobs["Skills"].apply(lambda text: remove_punctuation(text))
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "J":wordnet.ADJ, "R":wordnet.ADV}
jobs["Skills"] = jobs["Skills"].apply(lambda text: lemmatize_words(text,lemmatizer,wordnet_map))
jobs["Skills"] = jobs["Skills"].apply(lambda text: remove_escape(text))
jobs["Skills"]=jobs.Skills.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in STOPWORDS) )
jobs["Location"]=jobs.Location.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in STOPWORDS) )
jobs['JobID'] = jobs.index

# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
# tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(jobs.Description)]
# model_d2v = Doc2Vec(alpha=0.025, min_count=1)
  
# model_d2v.build_vocab(tagged_data)

# for epoch in range(10):
#     model_d2v.train(tagged_data,
#                 total_examples=model_d2v.corpus_count,
#                 epochs=model_d2v.epochs)
    
# document_embeddings=np.zeros((jobs.shape[0],len(model_d2v.docvecs[0])))

# for i in range(len(document_embeddings)):
#     document_embeddings[i]=model_d2v.docvecs[i]
    
# pairwise_similarities=cosine_similarity(document_embeddings)

model = TfidfVectorizer()
tfidf_matrix=model.fit_transform(jobs['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
array_one =np.ones((1059,1059)) 
distance = np.subtract(array_one,cosine_sim)
distance[distance <0] =0
distance = distance*0.45


model1 = TfidfVectorizer()
tfidf_matrix=model1.fit_transform(jobs['Skills'])
cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
distance_skills = np.subtract(array_one,cosine_sim1)
distance_skills[distance_skills <0] =0
distance_skills = distance_skills*.45

# new_distance = np.add(distance,distance_skills)
# new_distance = new_distance/2

model2 = TfidfVectorizer()
tfidf_matrix=model2.fit_transform(jobs['Location'])
cosine_sim2 = cosine_similarity(tfidf_matrix, tfidf_matrix)
distance_location = np.subtract(array_one,cosine_sim1)
distance_location[distance_location <0] =0
distance_location = distance_location*.1

new_distance = np.add(distance,distance_skills,distance_location)


distance1=distance
distance= new_distance
skills =input("Enter your skills:")
location = input("Enter your location:")
skills_df = pd.DataFrame({'Skills':[skills+" "+location]})
skills_df["Skills"]=skills_df.Skills.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]','',w).lower() not in STOPWORDS) )
skills_df = pd.concat([jobs['Skills']+jobs['Location'],skills_df['Skills']],axis=0,ignore_index=True)
skills_df = pd.DataFrame(skills_df)
tf = TfidfVectorizer()
tf_matrix=tf.fit_transform(skills_df[0])
sim=cosine_similarity(tf_matrix, tf_matrix)
score = list(sim[1059])
s = np.array(score)
sort_index = np.argsort(s)[::-1]
source = sort_index[1]
predicted = GBA(distance,source)
predicted_title = list(jobs[jobs['JobID'].isin(predicted)].Title)
predicted_location = list(jobs[jobs['JobID'].isin(predicted)].Location)
 

 
# Evaluation metrics 


actual_user_history = pd.read_csv('Job.csv')
actual_user_history = actual_user_history.drop(['Timestamp'],axis=1)
user_id=[0,1,2,3,4,5]

avg_precision=0

for i in user_id:
    user =i
    actual_user_input = list(actual_user_history.loc[user].values)
    skills =actual_user_input[0] +" "+actual_user_input[1]
    skills_df = pd.DataFrame({'Skills':[skills]})
    skills_df["Skills"]=skills_df.Skills.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]','',w).lower() not in STOPWORDS) )
    skills_df = pd.concat([jobs['Skills']+" "+jobs['Location'],skills_df['Skills']],axis=0,ignore_index=True)
    skills_df = pd.DataFrame(skills_df)
    tf = TfidfVectorizer()
    tf_matrix=tf.fit_transform(skills_df[0])
    sim=cosine_similarity(tf_matrix, tf_matrix)
    score = list(sim[1059])
    s = np.array(score)
    sort_index = np.argsort(s)[::-1]
    source = sort_index[1]
    actual_title = actual_user_input[2:]
    actual = list(jobs[jobs['Title'].isin(actual_title)].JobID)
    predicted = GBA(distance,source)
    predicted_title = list(jobs[jobs['JobID'].isin(predicted)].Title)
    correct = 0
    count=0
    sum_pre =0
    precision_l=[]
    recall_l=[]
    for pred in predicted_title:
        if pred in actual_title:
            correct = correct +1
        if count==5:
            break    
        count=count+1
        if len(predicted)!=0:
            precision =  correct/count
            sum_pre = precision+sum_pre
            recall = correct/5
            precision_l.append(precision)
            recall_l.append(recall)            
    ap =(1/count)*sum_pre
    avg_precision = avg_precision + ap
    print("Average precision of user",i, ":" ,ap)
    
    fig, ax = plt.subplots()
    ax.plot(recall_l, precision_l, markersize=10, marker="o")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall for AP@5")
    

print("Mean Average Precision:",avg_precision/len(user_id))
    
    

