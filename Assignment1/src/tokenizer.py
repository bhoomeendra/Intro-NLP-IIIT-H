#!/usr/bin/python
# coding=utf-8

import re
import yaml
from tqdm import tqdm

class Tokenizer():

    def __init__(self,doc_path):
        
        self.pre_embedding_path = '../../Embedding/glove.6B.50d.txt'
        self.doc_path = doc_path
        self.regex_rules ={
        16:[r'(\w+)-(\w+)',r'\1\2'],
        17:[r'([a-z]+)\'s', r'\1 is'],    # Replacing that's to that is
        18:[r'([i])\'m', r'\1 am'],       # Replaceing i'm to i am
        19:[r'([a-z]+)\'ve', r'\1 have'], # Replacing we've, i've to we have and i have
        20:[r'([a-z]+)\'d', r'\1 had'],   # Replacing i'd, they'd. to i had and they had
        21:[r'([a-z]+)\'ll', r'\1 will'], # Replacing i'll, they'll. to i will and they will
        22:[r'([a-z]+)\'re', r'\1 are'],  # Replacing we're, they're to we are and they are.
        23:[r'([a-z]+)in\'', r'\1ing'],   # Replacing tryin', doin' to tyring and doing
        24:[r'\n+',' '],
        25:[r'\t+', ' '],
        26:[r'can\'t', r'can not'],
        28:[r'won\'t', r'will not'],
        29:[r'([a-z]+)n\'t',r'\1 not'],   # Replacing couldn't with could not
        30:[r"@\w+",' <HASHTAG> '],
        31:[r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r' <URL> '],
        32:[r"#\w+",' <MENTION> '],       # URL might have # in them
        39:['[!"#＄%&\'()*+,-./:;=?@[]\^_`{\|}~]',' '], # Remove all puncutations
        40:[r'[^ -~]',' '],
        41:[r"\s+",' ']
        }
        self.order =  list(self.regex_rules.keys())
        self.order.sort()
        self.vocab = self.vocab_constructive()
        

    def get_pretrained_vocab(self):

    	fi = open(self.pre_embedding_path,'rt')
    	return set([ x.split(' ')[0] for x in fi.read().strip().split('\n')])
    

    def sentence_maker(self,text):

    	return text.split('\n\n')


    def sent2token(self,text,print=False):

        sent = self.clean(text)
        if len(sent) == 0:
            return None
        tokens = []
        for word in sent.split():
            if word in set(self.vocab):
                tokens.append(self.word_idx[word])
            else:
                tokens.append(self.word_idx['[UNK]'])
        return tokens


    def token2sent(self,tokens):
    	
    	out = ''
    	for idx in tokens:
    		out += self.idx_word[idx] + ' '
    	return out.strip()

    
    def vocab_constructive(self):

        out = open(self.doc_path,'r').read()
        clean = self.clean(out,st_end=True)
        words = set()

        for sent in tqdm(self.sentence_maker(clean),desc ='Vocab constuction'):
            for word in sent.split():
                words.add(word)

        pre_trained =  self.get_pretrained_vocab()
        vocab = list(words.intersection(pre_trained))
        vocab.insert(0, '[PAD]')
        vocab.insert(1,'[UNK]')
        vocab.insert(2,'[STR]')
        vocab.insert(3,'[END]')

        self.word_idx = { v:k for k,v in enumerate(vocab)}
        self.idx_word = { v:k for k,v in self.word_idx.items()}

        return vocab


    def clean(self, text,st_end = False):

        text  = text.lower()
        for rule in tqdm(self.order,desc='Cleaning:',disable=True):
            patter,sub_text = self.regex_rules[rule]
            text = re.sub(patter,sub_text,text)
        
        if st_end == False:
            text = self.idx_word[2]+' '+ text +' '+self.idx_word[3] 
        return text.strip()



if __name__=='__main__':
	
	token = Tokenizer(doc_path='../data/Pride and Prejudice - Jane Austen.txt')

	print(token.sent2token("This is a okay sentence"))
	print(token.token2sent([324,34,4,76,78,1,2,0,3]))