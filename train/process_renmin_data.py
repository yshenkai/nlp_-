#coding=gbk
import numpy as np
import collections
from keras.preprocessing.sequence import pad_sequences
def _read_data(f):
    split_text='\n'
    fields=f.read().decode('utf-8')
    data = [[row.split(" ") for row in sample.split(split_text)] for
            sample in
            fields.strip().split(split_text + split_text)]
    f.close()
    return data
def _load_vocab():
    vocab=collections.OrderedDict()
    f=open("../data/vocab.txt",'rb').readlines()
    index=0
    for line in f:
        vocab[line.decode('utf-8').strip()]=index
        index+=1
    return vocab
label=["B-ORG","I-ORG"]

def _process_data(data,vocab,label_tag,max_len=1000,onehot=True):
    if max_len is None:
        max_len=max(len(s) for s in data)
    print("max_len",max_len)
    data2id=[[vocab.get(w[0].lower(),0) for w in line] for line in data]
    label2id=[[1 if w[1] in label else 0 for w in line] for line in data]
    # label2id=[]
    # for l in label_tag:
    #     if l in label:
    #         label2id.append(1)
    #     else:
    #         label2id.append(0)
    data2id=pad_sequences(data2id,max_len)
    label2id=pad_sequences(label2id,max_len,value=-1)
    if onehot:
        label2id=np.eye(len(label_tag),dtype="float32")[label2id]
    else:
        label2id=np.expand_dims(label2id,2)
    return data2id,label2id


def load_data():
    train=_read_data(open("../data_constract_xunlian/data_xunlian/chandi/train_data.data",'rb'))
    #test=_read_data(open("../data/test_data.data",'rb'))
    label_tag=[0,1]

    vocab=_load_vocab()
    train=_process_data(train,vocab,label_tag)
    #test=_process_data(test,vocab,label_tag)
    print("load_data_data2id.shape",train[0].shape)
    print("load_data_label2id.shape", train[1].shape)
    return train,(vocab,label_tag)

if __name__=="__main__":
    train,(vocab,label_tag)=load_data()
    print(train[0].shape)
    print(train[1].shape)
    print(train[0][0])
    print(train[1][0])