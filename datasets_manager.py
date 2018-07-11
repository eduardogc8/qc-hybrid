# -*- coding: utf-8 -*-
import xml.etree.ElementTree as et
import util
import re

chave_path_questions = "data/datasets/Chave.xml"
uiuc_path_train_questions = "data/datasets/UIUC/train_5500.label.txt"
uiuc_path_test_questions = "data/datasets/UIUC/TREC_10.label.txt"
uiuc_pt_path_train_questions = "data/datasets/UIUC_PT/train.txt"
uiuc_pt_path_test_questions = "data/datasets/UIUC_PT/test_.txt"


# Open xml file and return it tree
def questions_tree(path):
    tree = et.parse(path)
    return tree.getroot()


# Return a list with all Chave questions as dictionary
def chave_questions():
    ret = []
    count = 1
    for question in questions_tree(chave_path_questions):
        q = {}
        q['id'] = count
        q['id_org'] = question.attrib['id_org']
        q['year'] = question.attrib['ano']
        q['category'] = question.attrib['categoria']
        q['type'] = question.attrib['tipo']
        q['class'] = chave_pair_classification(q['category'], q['type'])
        q['ling'] = question.attrib['ling_orig']
        if u'restrição' in question.attrib:
            q['restriction'] = question.attrib[u'restrição']
        if 'restricao' in question.attrib:
            q['restriction'] = question.attrib['restricao']
        if q['restriction'] == 'NO':
            q['restriction'] = 'NONE'
        if q['restriction'] == 'X':
            q['restriction'] = ''

        q['answers'] = []
        q['extracts'] = []

        for e in question:
            if e.tag == 'texto':
                q['question'] = e.text
            if e.tag == 'resposta':
                if e.text is None or util.treat_text(e.text) == '':
                    continue
                ans = {'answer': e.text, 'n': e.attrib['n'], 'doc': e.attrib['docid']}
                ans['doc'] = chave_validate_docid(ans['doc'])
                q['answers'].append(ans)
            if e.tag == 'extracto':
                q['extracts'].append({'extract': e.text, 'n': e.attrib['n'], 'answer_n': e.attrib['resposta_n']})
        count += 1
        ret.append(q)
    return ret


# Check if a docid is valid (FolhaSP or Público) so return a docid in the
# documents docid else return None
def chave_validate_docid(docid):
    if docid is None or len(docid.strip()) < 10:
        return None
    else:
        docid = docid.strip()
        r = re.match(r'(?P<n1>.+)(?P<n2>\d{6,8})-(?P<n3>\d{1,3}$)', docid)
        if r is not None:
            if r.groupdict()['n1'][:1] == 'F':
                return 'FSP'+r.groupdict()['n2']+'-'+r.groupdict()['n3']
            else:
                return 'PUBLICO-19'+r.groupdict()['n2']+'-'+r.groupdict()['n3']
        else:
            return None


# Return the right question class based in category and type attributes
def chave_pair_classification(c, t):
    if c == 'COUNT':
        return 'MEASURE'
    if c == 'D' or c == 'DEFINITION':
        return 'DEFINITION'
    if c == 'F' or c == 'FACTOID':
        if t == 'COUNT':
            return 'MEASURE'
        else:
            return t
    if c == 'L' or c == 'LIST':
        if t == 'COUNT':
            return 'MEASURE'
        else:
            return t
    if c == 'LOCATION':
        return 'LOCATION'
    if c == 'MEASURE':
        return 'MEASURE'
    if c == 'OBJECT':
        return 'DEFINITION'
    if c == 'ORGANIZATION':
        return 'ORGANIZATION'
    if c == 'OTHER' and (t == 'FACTOID' or t == 'LIST'):
        return 'OTHER'
    if c == 'OTHER' and not (t == 'FACTOID' or t == 'LIST'):
        return t
    if c == 'PERSON' and t == 'DEFINITION':
        return 'DEFINITION'
    if c == 'PERSON' and not t == 'DEFINITION':
        return 'PERSON'
    if c == 'TIME':
        return 'TIME'
    return c


def uiuc_questions():
    train = []
    lines = open(uiuc_path_train_questions).readlines()
    for line_ in lines:
        q = {}
        line = line_.strip()
        q['class'] = line[:line.index(':')]
        q['sub_class'] = line[line.index(':')+1:line.index(' ')]
        q['question'] = line[line.index(' '):].strip()
        train.append(q)

    test = []
    lines = open(uiuc_path_test_questions).readlines()
    for line_ in lines:
        q = {}
        line = line_.strip()
        q['class'] = line[:line.index(':')]
        q['sub_class'] = line[line.index(':')+1:line.index(' ')]
        q['question'] = line[line.index(' '):].strip()
        test.append(q)

    return train, test


def uiuc_pt_questions():
    train = []
    lines = open(uiuc_pt_path_train_questions, encoding='utf-8').readlines()
    for line_ in lines:
        q = {}
        line = line_.strip()
        q['class'] = line[:line.index(':')]
        q['sub_class'] = line[line.index(':')+1:line.index(' ')]
        q['question'] = line[line.index(' '):].strip()
        train.append(q)

    test = []
    lines = open(uiuc_pt_path_test_questions).readlines()
    for line_ in lines:
        q = {}
        line = line_.strip()
        q['class'] = line[:line.index(':')]
        q['sub_class'] = line[line.index(':')+1:line.index(' ')]
        q['question'] = line[line.index(' '):].strip()
        test.append(q)

    return train, test
