from unicodedata import normalize
import io


CHARACTERS_REPLACE = [('\n',''), ('\\', ''), (u'«', "\""), (u'»', "\""), (u"'", "\"")]
PONCTUATION_REPLACE = [(';', ''), (':', ''), ('(', ''), (')', ''), ('[', ''), (']', ''), ('{', ''), ('}', ''), ('?', ''), ('!', '')]
NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

PATH_STOPWORDS = 'data/util/stopwords_pt.txt'


#Retorna True se a string representa um INT
def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


#Remove o acento das palavras
def remove_acentts(txt):
    #if type(txt) is str:
    #    txt = unicode(txt, 'utf-8')
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')


# Tratamento no texto
def treat_text(text):
    for replace in CHARACTERS_REPLACE:
        text = text.replace(replace[0], replace[1])
    text = text.strip()
    text = " ".join(text.split())
    return text


# Remove os caracteres conforme PONCTUATION_REPLACE do texto
def replace_ponctutation(text):
    for replace in PONCTUATION_REPLACE:
        text = text.replace(replace[0], replace[1])
    if ',' in text or '.' in text:
        new_text = ''
        for i in range(len(text)):
            c = text[i]
            if c == ',' or c == '.':
                if i > 0 and text[i-1] in NUMBERS and i < len(text)-1 and text[i+1] in NUMBERS:
                    new_text += c
            else:
                new_text += c
        text = new_text
    text = text.strip()
    return text


def is_stopword(word):
    f = io.open(PATH_STOPWORDS, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        if word.lower() == treat_text(line).lower():
            return True
    return False


def findSentenceIndexs(text, sentence):
    indexs = []
    sentIndex = 0
    for i in range(len(text.split())):
        word = text.split()[i]
        if word == sentence.split()[sentIndex]:
            if sentIndex == len(sentence.split())-1:
                indexs.append((i-sentIndex, i))
                sentIndex = 0
            else:
                sentIndex += 1
        else:
            sentIndex = 0
    return indexs


def shortSentenceDistance(text, word, answer):
    if answer not in text:
        print(answer + ' | ' + text)
        return -2
    if word in text.split():
        indexs1 = findSentenceIndexs(text, word)
        indexs2 = findSentenceIndexs(text, word)
        if len(indexs1) <= 0:
            print('Index1 '+word+' : ' + text)
        if len(indexs2) <= 0:
            print('Index2 '+answer+' : ' + text)
        distance = float('inf')
        for i1 in indexs1:
            for i2 in indexs2:
                if i1[0] > i2[0]:
                    dist = abs(i1[0] - i2[1])
                else:
                    dist = abs(i1[1] - i2[0])
                if dist < distance:
                    distance = dist
        return distance
    return -1

def myHotEncode(input_data, max_vocab=0, vocab2idx=None):
    "Return the hot-vecotor and the vocab2idx."

    import numpy as np
    from collections import OrderedDict
    from operator import itemgetter

    if vocab2idx is None:
        vocabFreq = {}
        for i in input_data:
            for j in i:
                if j not in vocabFreq:
                    vocabFreq[j] = 0
                vocabFreq[j] += 1
        vocabFreq = OrderedDict(sorted(vocabFreq.items(), key=itemgetter(1), reverse=True))
        vocab2idx = {}
        count = 0
        for v in vocabFreq:
            count += 1
            if max_vocab > 0 and count > max_vocab:
                break
            vocab2idx[v] = len(vocab2idx)
    vocabEmbeddings = np.identity(len(vocab2idx), dtype='float32')
    data_ret = []
    for i in input_data:
        i_ = []
        for j in i:
            if j in vocab2idx:
                i_.append(vocabEmbeddings[vocab2idx[j]])
        if len(i_) == 0:
            i_ = np.zeros((1, len(vocab2idx)))
        data_ret.append(i_)
    return data_ret, vocab2idx


def myHotDecode(input_data, vocab2idx):
    "Return the decode as final representation and decode as indexs"

    data_ = []
    data_idx = []
    for i in input_data:
        i_ = []
        i_idx = []
        if len(i) != len(vocab2idx):
            print('Erro:', 'The vocab2idx not fit the input data!')
            return
        for _i, j in enumerate(i):
            if j > 0:
                v = [k for k in vocab2idx if vocab2idx[k]==_i][0]
                i_.append(v)
                i_idx.append(_i)
        data_.append(i_)
        data_idx.append(i_idx)
    return data_, data_idx


def text_to_wordEmbedding(text, wordEmbedding):
    import nltk
    import numpy as np
    if type(text) != str:
        ret = []
        for text_ in text:
            text_ = str(text_)
            in_tokens = np.array([token for token in nltk.tokenize.word_tokenize(text_) if token in wordEmbedding])
            if len(in_tokens) > 0:
                ret.append(sum(wordEmbedding[in_tokens]))
            else:
                ret.append(np.zeros(wordEmbedding.vector_size))
        return ret
    in_tokens = [token for token in nltk.tokenize.word_tokenize(text) if token in wordEmbedding]
    if len(in_tokens) > 0:
        return sum(wordEmbedding[in_tokens])
    else:
        return np.zeros(wordEmbedding.vector_size)
