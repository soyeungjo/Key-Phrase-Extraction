# https://github.com/yutayamazaki/EmbedRank/blob/master/embedrank/embedrank.py
#%%
import os
import json
import time

import pandas as pd
import numpy as np

# !pip install gensim==3.8.3
import gensim
gensim.__version__

from tqdm import tqdm


#%%
path = 'C:/Users/SOYOUNG/Desktop/001.문서요약텍스트_sample/원시데이터/신문기사/sample_original.json'


with open(path, 'r', encoding = 'UTF8') as json_file:
    json_data = json.load(json_file)

train_df = pd.DataFrame(json_data)
train_df.shape
train_df.columns


text_df = pd.DataFrame({'title': train_df['title']})
text_df['text'] = ['.'.join([train_df['text'][j][i]['sentence'] for i in range(len(train_df['text'][j]))]) 
                   for j in range(train_df.shape[0])]
text_df['news'] = text_df['title'] + '. ' + text_df['text']
text_df['extractive'] = [' '.join([train_df['text'][j][i]['sentence'] for i in train_df['extractive'][j]]) 
                         for j in range(train_df.shape[0])]

'''
    title: 제목
    text: 내용
    news: 제목 + 내용
    extractive: reference
'''


#%%
from konlpy.tag import Hannanum


hannanum = Hannanum()

def hannanum_phrase_gen(doc):
    postag = hannanum.pos(doc)
    
    for i in range(len(postag)):
        postag[i] = list(postag[i])
        
        if ('J' in postag[i] or 'E' in postag[i]):
            postag[i-1][0] = postag[i-1][0] + postag[i][0] 
        
    postag = [pos for pos in postag if 'J' not in pos and 'E' not in pos]
    
    index = []
    words = []
    
    for i in range(len(postag)):
        if ('N' in postag[i] or 'P' in postag[i]):
            index.append(i)
            words.append(postag[i][0])
    
    ex_df = pd.DataFrame({'index': index, 'words': words})
    
    phrase = []
    phrase_idx = ex_df['index'][1:].reset_index(drop = True) - ex_df['index'][:-1].reset_index(drop = True)
    cut_idx = np.where(phrase_idx != 1)[0] + 1
    
    for i in range(len(cut_idx)):
        if i == 0:
            phrase.append(' '.join(ex_df['words'][:cut_idx[i]]))
        else:
            phrase.append(' '.join(ex_df['words'][cut_idx[i-1]:cut_idx[i]]))
    
    return(phrase)

# p = phrase_gen(text_df['text'][0])
# [hannanum.morphs(p[i]) for i in range(len(p))]



#%%
hannanum_tokenized_doc = [hannanum.morphs(text_df['news'][i]) for i in tqdm(range(text_df.shape[0]))]

# all_sentences = [[train_df['text'][j][i]['sentence'] for i in range(len(train_df['text'][j]))] 
#                  for j in range(text_df.shape[0])]
# tokenized_sent = [[okt.morphs(all_sentences[j][i]) for i in range(len(all_sentences[j]))] 
#                   for j in tqdm(range(text_df.shape[0]))]

def hannanum_tokenized_phrase_func(phrase):
    token = [hannanum.morphs(phrase[i]) for i in range(len(phrase))]
    return(token)


#%%
def cosim(u, v):
    return(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


#%%
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


hannanum_tagged_doc = [TaggedDocument(d, [i]) for i, d in enumerate(hannanum_tokenized_doc)]

start = time.time()
model = Doc2Vec(hannanum_tagged_doc, vector_size = 700, window = 2, min_count = 1, epochs = 3000)
print(time.time() - start)

# model.build_vocab
# model.docvecs.doctags.keys()

start = time.time()
model.train(hannanum_tagged_doc, total_examples = model.corpus_count, epochs = model.epochs)
print(time.time() - start)
model.wv.vocab

model.save('C:/Users/SOYOUNG/Desktop/001.문서요약텍스트_sample/hannanum_d2v.model')
# hannanum_model = Doc2Vec.load('C:/Users/SOYOUNG/Desktop/001.문서요약텍스트_sample/hannanum_d2v.model')


#%%
d = 233

hannanum_phrase = hannanum_phrase_gen(text_df['news'][d])
hannanum_tokenized_phrase = hannanum_tokenized_phrase_func(hannanum_phrase)

hannanum_embed_vec = model[d]
hannanum_sent_embed = [model.infer_vector(i) for i in hannanum_tokenized_phrase]
hannanum_embed_vec = np.append([hannanum_embed_vec], hannanum_sent_embed, axis = 0)

hannanum_df = pd.DataFrame({'tag': ['doc']+['phrase{}'.format(i) for i in range(len(hannanum_sent_embed))],
                            'text': [text_df['news'][d]] + hannanum_phrase,
                            'cosim': [cosim(hannanum_embed_vec[0], hannanum_embed_vec[i]) 
                                      for i in range(len(hannanum_sent_embed)+1)]})
hannanum_df['cosim_rank'] = hannanum_df['cosim'].rank(method = 'dense', ascending = False)


#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_x = pca.fit_transform(hannanum_embed_vec)

hannanum_df['x'] = pca_x[:, 0]
hannanum_df['y'] = pca_x[:, 1]


#%%
k = 10

plt.figure(figsize=(8, 8))
plt.scatter(hannanum_df['x'][0], hannanum_df['y'][0], marker = '*', color = 'blue', s = 300)
plt.text(hannanum_df['x'][0], hannanum_df['y'][0] + 0.1, 'document', fontsize = 10)
for i in range(1, hannanum_df.shape[0]):
    if hannanum_df['cosim_rank'][i] <= k+1:
        plt.scatter(hannanum_df['x'][i], hannanum_df['y'][i], color = 'blue', s = 100)
        plt.text(hannanum_df['x'][i], hannanum_df['y'][i] + 0.1, hannanum_df['tag'][i], fontsize = 10)
    else:
        plt.scatter(hannanum_df['x'][i], hannanum_df['y'][i], facecolor = 'none', edgecolors = 'blue', s = 100)
        plt.text(hannanum_df['x'][i], hannanum_df['y'][i] + 0.1, hannanum_df['tag'][i], fontsize = 10)    
plt.show()
plt.close()


#%%
# train_df['extractive'][d]
text_df['extractive'][d]

hannanum_df.nsmallest(k + 1, 'cosim_rank')['text'][1:].values





#%%
from konlpy.tag import Okt

okt = Okt()

def okt_phrase_gen(doc):
    postag = okt.pos(doc)
    
    for i in range(len(postag)):
        postag[i] = list(postag[i])
        
        if ('Suffix' in postag[i] and 'Josa' in postag[i+1]):
            postag[i-1][0] = postag[i-1][0] + postag[i][0] + postag[i+1][0]
        
        elif ('Josa' in postag[i] or 'Suffix' in postag[i]):
            postag[i-1][0] = postag[i-1][0] + postag[i][0] 
            
        
    postag = [pos for pos in postag if 'Josa' not in pos and 'Suffix' not in pos]
    
    index = []
    words = []
    
    for i in range(len(postag)):
        if ('Noun' in postag[i] or 'Adjective' in postag[i]):
            index.append(i)
            words.append(postag[i][0])
    
    ex_df = pd.DataFrame({'index': index, 'words': words})
    
    phrase = []
    phrase_idx = ex_df['index'][1:].reset_index(drop = True) - ex_df['index'][:-1].reset_index(drop = True)
    cut_idx = np.where(phrase_idx != 1)[0] + 1
    
    for i in range(len(cut_idx)):
        if i == 0:
            phrase.append(' '.join(ex_df['words'][:cut_idx[i]]))
        else:
            phrase.append(' '.join(ex_df['words'][cut_idx[i-1]:cut_idx[i]]))
    
    return(phrase)

# p = phrase_gen(text_df['text'][0])
# [hannanum.morphs(p[i]) for i in range(len(p))]



#%%
okt_tokenized_doc = [okt.morphs(text_df['news'][i]) for i in tqdm(range(text_df.shape[0]))]

# all_sentences = [[train_df['text'][j][i]['sentence'] for i in range(len(train_df['text'][j]))] 
#                  for j in range(text_df.shape[0])]
# tokenized_sent = [[okt.morphs(all_sentences[j][i]) for i in range(len(all_sentences[j]))] 
#                   for j in tqdm(range(text_df.shape[0]))]

def okt_tokenized_phrase_func(phrase):
    token = [okt.morphs(phrase[i]) for i in range(len(phrase))]
    return(token)


#%%
def cosim(u, v):
    return(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))


#%%
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


okt_tagged_doc = [TaggedDocument(d, [i]) for i, d in enumerate(okt_tokenized_doc)]

start = time.time()
okt_model = Doc2Vec(okt_tagged_doc, vector_size = 700, window = 2, min_count = 1, epochs = 3000)
print(time.time() - start)

# model.build_vocab
# model.docvecs.doctags.keys()

start = time.time()
okt_model.train(okt_tagged_doc, total_examples = okt_model.corpus_count, epochs = okt_model.epochs)
print(time.time() - start)
okt_model.wv.vocab

okt_model.save('C:/Users/SOYOUNG/Desktop/001.문서요약텍스트_sample/okt_d2v.model')
# okt_model = Doc2Vec.load('C:/Users/SOYOUNG/Desktop/001.문서요약텍스트_sample/hannanum_d2v.model')


#%%
d = 233

okt_phrase = okt_phrase_gen(text_df['news'][d])
okt_tokenized_phrase = okt_tokenized_phrase_func(okt_phrase)

okt_embed_vec = okt_model[d]
okt_sent_embed = [okt_model.infer_vector(i) for i in okt_tokenized_phrase]
okt_embed_vec = np.append([okt_embed_vec], okt_sent_embed, axis = 0)

okt_df = pd.DataFrame({'tag': ['doc']+['phrase{}'.format(i) for i in range(len(okt_sent_embed))],
                            'text': [text_df['news'][d]] + okt_phrase,
                            'cosim': [cosim(okt_embed_vec[0], okt_embed_vec[i]) 
                                      for i in range(len(okt_sent_embed)+1)]})
okt_df['cosim_rank'] = okt_df['cosim'].rank(method = 'dense', ascending = False)


#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_x = pca.fit_transform(okt_embed_vec)

okt_df['x'] = pca_x[:, 0]
okt_df['y'] = pca_x[:, 1]


#%%
k = 10

plt.figure(figsize=(8, 8))
plt.scatter(okt_df['x'][0], okt_df['y'][0], marker = '*', color = 'blue', s = 300)
plt.text(okt_df['x'][0], okt_df['y'][0] + 0.1, 'document', fontsize = 10)
for i in range(1, okt_df.shape[0]):
    if okt_df['cosim_rank'][i] <= k+1:
        plt.scatter(okt_df['x'][i], okt_df['y'][i], color = 'blue', s = 100)
        plt.text(okt_df['x'][i], okt_df['y'][i] + 0.1, okt_df['tag'][i], fontsize = 10)
    else:
        plt.scatter(okt_df['x'][i], okt_df['y'][i], facecolor = 'none', edgecolors = 'blue', s = 100)
        plt.text(okt_df['x'][i], okt_df['y'][i] + 0.1, okt_df['tag'][i], fontsize = 10)    
plt.show()
plt.close()


#%%
# train_df['extractive'][d]
text_df['extractive'][d]

okt_df.nsmallest(k + 1, 'cosim_rank')['text'][1:].values