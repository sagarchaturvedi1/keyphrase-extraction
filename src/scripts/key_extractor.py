from spacy.en import English
import pandas as pd
import numpy as np
import os
import re



def key_chunk(parsed, pattern):
    ''' This is an incomplete function - Not Required'''

    for sent in parsed.sents:
        signature = ' '.join(['<%s>' % w.tag_ for w in sent])

        m = re.search(pattern,signature)
        if m:
            print m.group(1)
            

def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    ''' This function will extract text of a specific POS sequence rather than just Noun Phrase '''

    import itertools, nltk, string
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group)
                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    ''' This is for calculating the TF-IDF of the extracted candidate keyphrases.'''

    import gensim, nltk
    
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf, dictionary

def extract_candidate_features(candidates, doc_text, doc_excerpt, doc_title):
    '''For creating features based on doc title, phrase length, g^2 statistics.
    This is a very simple feature file. Won't work for us as it is difficult to extract the title'''

    import collections, math, nltk, re
    
    candidate_scores = collections.OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    for candidate in candidates:
        
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # frequency-based
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        # count could be 0 for multiple reasons; shit happens in a simplified example
        if not cand_doc_count:
            print '**WARNING:', candidate, 'not found!'
            continue
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # positional
        # found in title, key excerpt
        in_title = 1 if pattern.search(doc_title) else 0
        in_excerpt = 1 if pattern.search(doc_excerpt) else 0
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        abs_first_occurrence = first_match.start() / doc_text_length
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(doc_text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length
            spread = abs_last_occurrence - abs_first_occurrence

        candidate_scores[candidate] = {'term_count': cand_doc_count,
                                       'term_length': term_length, 'max_word_length': max_word_length,
                                       'spread': spread, 'lexical_cohesion': lexical_cohesion,
                                       'in_excerpt': in_excerpt, 'in_title': in_title,
                                       'abs_first_occurrence': abs_first_occurrence,
                                       'abs_last_occurrence': abs_last_occurrence}

    return candidate_scores

def main():

    parser = English()
    train_dir = "C:/Users/sagar/Dropbox/CourseWork/Structured Prediction/Project/work/data/train"
    files = os.listdir(train_dir)
    texts = []

    for file in files:
        if '.txt' in file:
            fp = open(os.path.join(train_dir,file), 'r')
            text = fp.read()
            parsed = parser(unicode(text, "utf-8"))
            texts.append(unicode(text, 'utf-8'))
            
        if '.ann' in file:
            ann = pd.read_table(os.path.join(train_dir,file), names = ['TagId', 'Label', 'Phrase'], delimiter='\t')
            ann = ann[ann.TagId != '*']

            


    print ann

    ''' Printing sentences and Spacy noun chunks
    for i, s in enumerate(parsed.sents):
        print i,s    

    for i, n in enumerate(parsed.noun_chunks):
        print i,n

    #pattern = re.compile(r'(<JJ>* (<NN>|<NNS>|<NNP>)+ <IN>)? (<JJ>* (<NN>|<NNS>|<NNP>)+)')
    #pattern = re.compile(r'((<JJ>)* (<NN.?>)+ <IN>)? (<JJ>)* (<NN.?>)+')
    #key_chunk(parsed, pattern)
    '''

    cand_chunks = extract_candidate_chunks(unicode(text, 'utf-8'))

    for cand in cand_chunks:
        print cand

    print file
    ''' Calculating the TF-IDF
    corpus_tfidf, dictionary = score_keyphrases_by_tfidf(texts)
    print '\n', file, '\n'
    for key in dictionary:
        print key, dictionary[key]
    '''

if __name__ == "__main__":
    main() 
