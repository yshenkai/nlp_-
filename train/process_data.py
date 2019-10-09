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
    str = "潍坊市食品药品监督管理局通过其官方网站公布了“2016年3月份流通环节食品市级监督抽检结果的通告”。此次抽检品种为肉制品、蔬菜制品、调味品、食用农产品等13个种类，经检测发现不合格样品12批次，抽检合格率为98.83%。其中，山东临朐百货大楼有限公司龙泉路广场所售茌平县恒源食品有限公司生产的金针菇，因二氧化硫含量超标，遭到通报。近日，潍坊市食品药品监督管理局通过其官方网站公布了“2016年3月份流通环节食品市级监督抽检结果的通告”。据悉，潍坊市食品药品监督管理局3月份抽检品种为肉制品、蔬菜制品、调味品、食用农产品等13个种类，以非食用物质、药残留、重金属、食品添加剂等易出现问题的项目为检测重点，共计700批次。经检测发现不合格样品12批次，抽检合格率为98.83%。其中，山东临朐百货大楼有限公司龙泉路广场所售茌平县恒源食品有限公司生产的金针菇，因二氧化硫含量超标，遭到通报。通报如下：被抽检产品为金针菇（香辣），被抽检单位为山东临朐百货大楼有限公司龙泉路广场，标称生产企业为茌平县恒源食品有限公司，规格为175g/瓶，生产日期为2015/11/8，不合格项目为二氧化硫。对上述抽检中发现的不合格产品，抽样场所所在地县级食品药品监管部门（市场监督管理部门）按照《中华人民共和国》的规定，依法进行处理。标称市外生产的不合格产品信息已通报相关食品生产企业所在地食品药品监督管理局。"
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
    # print(voc.get("大",0))
    # _gen_test_data()
    train,(vocab, label_tag) = load_data()
    print(train[0])
    # train = _read_data(open("../data_constract_xunlian/data_xunlian/chandi/train_data_line_wise.data", 'rb'))
    # for i in train:
    #     print(i)