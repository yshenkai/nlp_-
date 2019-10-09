#coding=gbk
import numpy as np
import collections
from keras.preprocessing.sequence import pad_sequences

def _read_data(f):
    split_text='\r\n'
    fields=f.read().decode('utf-8')
    data = [[row.split("\t") for row in sample.split(split_text)] for
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
def _process_data(data,vocab,label_tag,max_len=256,onehot=True):
    if max_len is None:
        max_len=max(len(s) for s in data)
    print("max_len",max_len)
    # data2id=[[vocab.get(w[0].lower(),0) for w in line] for line in data]
    # label2id=[[int(w[1]) for w in line] for line in data]
    data2id=[]
    label2id=[]
    for line in data:
        d=[]
        l=[]
        for w in line:
            if(len(w)==2):
                d.append(vocab.get(w[0].lower(),0))
                l.append(int(w[1]))
        data2id.append(d)
        label2id.append(l)

    data2id=pad_sequences(data2id,max_len)
    label2id=pad_sequences(label2id,max_len,value=-1)
    if onehot:
        label2id=np.eye(len(label_tag),dtype="float32")[label2id]
    else:
        label2id=np.expand_dims(label2id,2)
    print("label_toid.shape",label2id.shape)
    return data2id,label2id




def load_data():
    train=_read_data(open("../data_constract_xunlian/data_xunlian/chandi/train_data_line_wise.data",'rb'))
    #test=_read_data(open("../data/test_data.data",'rb'))
    label_tag=[0,1]
    vocab=_load_vocab()
    train=_process_data(train,vocab,label_tag)
    #test=_process_data(test,vocab,label_tag)
    print("load_data_data2id.shape",train[0].shape)
    print("load_data_label2id.shape", train[1].shape)
    return train,(vocab,label_tag)


def _gen_test_data():
    str = "Ϋ����ʳƷҩƷ�ල�����ͨ����ٷ���վ�����ˡ�2016��3�·���ͨ����ʳƷ�м��ල�������ͨ�桱���˴γ��Ʒ��Ϊ����Ʒ���߲���Ʒ����ζƷ��ʳ��ũ��Ʒ��13�����࣬����ⷢ�ֲ��ϸ���Ʒ12���Σ����ϸ���Ϊ98.83%�����У�ɽ�����԰ٻ���¥���޹�˾��Ȫ·�㳡������ƽ�غ�ԴʳƷ���޹�˾�����Ľ��빽����������������꣬�⵽ͨ�������գ�Ϋ����ʳƷҩƷ�ල�����ͨ����ٷ���վ�����ˡ�2016��3�·���ͨ����ʳƷ�м��ල�������ͨ�桱����Ϥ��Ϋ����ʳƷҩƷ�ල�����3�·ݳ��Ʒ��Ϊ����Ʒ���߲���Ʒ����ζƷ��ʳ��ũ��Ʒ��13�����࣬�Է�ʳ�����ʡ�ҩ�������ؽ�����ʳƷ��Ӽ����׳����������ĿΪ����ص㣬����700���Ρ�����ⷢ�ֲ��ϸ���Ʒ12���Σ����ϸ���Ϊ98.83%�����У�ɽ�����԰ٻ���¥���޹�˾��Ȫ·�㳡������ƽ�غ�ԴʳƷ���޹�˾�����Ľ��빽����������������꣬�⵽ͨ����ͨ�����£�������ƷΪ���빽��������������쵥λΪɽ�����԰ٻ���¥���޹�˾��Ȫ·�㳡�����������ҵΪ��ƽ�غ�ԴʳƷ���޹�˾�����Ϊ175g/ƿ����������Ϊ2015/11/8�����ϸ���ĿΪ�������򡣶���������з��ֵĲ��ϸ��Ʒ�������������ڵ��ؼ�ʳƷҩƷ��ܲ��ţ��г��ල�����ţ����ա��л����񹲺͹����Ĺ涨���������д���������������Ĳ��ϸ��Ʒ��Ϣ��ͨ�����ʳƷ������ҵ���ڵ�ʳƷҩƷ�ල����֡�"
    vocab=_load_vocab()
    str2id=[[vocab.get(s,0) for s in str]]
    str2id=pad_sequences(str2id,maxlen=1000,padding='post')
    print("============",str2id)
    return str2id











if __name__=="__main__":
    #
    # label_tag = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    # data=_read_data(open("../data/test_data.data",'rb'))
    # vocab=_load_vocab()
    # train, test, (vocab, label_tag)=load_data()
    # print(test[0][0])
    # print(test[1][0])
    # load_data()
    # voc=_load_vocab()
    # print(voc.get("��",0))
    # _gen_test_data()
    train,(vocab, label_tag) = load_data()
    print(train[0])
    # train = _read_data(open("../data_constract_xunlian/data_xunlian/chandi/train_data_line_wise.data", 'rb'))
    # for i in train:
    #     print(i)