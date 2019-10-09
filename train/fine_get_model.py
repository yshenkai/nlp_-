from keras.layers import Dense,Embedding,Bidirectional,LSTM,Activation,Flatten,Dropout,TimeDistributed,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy,binary_crossentropy
from train import process_data
from keras import losses
from keras import metrics
from sklearn.metrics import recall_score,f1_score
import numpy as np
def get_model():
    (train_x, train_y), (vocab, label_tag) = process_data.load_data()
    print("train_x.shape",train_x.shape)
    print("train_y.shape",train_y.shape)
    model=Sequential()
    model.add(Embedding(len(vocab),256))
    model.add(Bidirectional(LSTM(256//2,return_sequences=True)))
    #model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(256//2,return_sequences=True)))
    # model.add(Dropout(0.5))
    model.add(Dense(2))
    #model.add(Activation("sigmoid"))
    #model.load_weights("../fine_train_output/jigou_recall_reduce_lr_epoch_weights.04-0.99.hdf5")
    #model.load_weights("../data_constract_xunlian/model_output/chandi/chandi_line_wise_recall_reduce_lr_epoch_weights.17-3.65.hdf5")
    model.compile(optimizer=Adam(lr=0.0001),loss=my_loss,metrics=[metrics.binary_accuracy,myprecision_sta,myrecall_sta])

    #model.fit(train_x,train_y,batch_size=2,epochs=1)
    return model,(train_x,train_y)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def myrecall_sta(y_true,y_pred):
    #y_pred=y_pred/0.4
    y_true=np.argmax(y_true,axis=-1)

    y_pred=np.argmax(y_pred,axis=-1)
    y_true=tf.convert_to_tensor(y_true,np.float32)
    y_pred=tf.convert_to_tensor(y_pred,np.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def myprecision_sta(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true=np.argmax(y_true,axis=-1)
    y_pred=np.argmax(y_pred,axis=-1)
    y_true=tf.convert_to_tensor(y_true,np.float32)
    y_pred=tf.convert_to_tensor(y_pred,np.float32)
    #y_pred=y_pred/0.4
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    precision =true_positives / (predicted_positives+K.epsilon())
    return precision

def mycrossentropy(y_true, y_pred, e=1):
    loss1 = binary_crossentropy(y_true, y_pred)
    loss2 = binary_crossentropy(K.ones_like(y_pred)/2, y_pred)
    return (10-e)*loss1 + e*loss2

def myloss(y_true,y_pred):
    return K.tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,25)
def myfocal_loss(y_true,y_pred):
    '''
    :param y_true:  [bacth_size,sql_len]
    :param y_pred:
    :return:
    '''

def focal_loss(labels, logits, gamma=1,at=0.25):
    #logits=tf.nn.sigmoid(logits)
    zeros=tf.zeros_like(logits,dtype=logits.dtype)
    poss_corr=tf.where(labels>zeros,labels-logits,zeros)
    neg_corr=tf.where(labels>zeros,zeros,logits)
    fl_loss=-(at)*(poss_corr**gamma)*tf.log(logits)-(1-at)*(neg_corr**gamma)*tf.log(1-logits)
    return tf.reduce_sum(fl_loss)
def focal_loss_1(labels,logits , gamma=2,at=0.4):
    labels=K.reshape(labels,[-1])
    logits=K.reshape(logits,[-1])
    fl_loss=-(at)*(labels**gamma)*tf.log(logits)-(1-at)*((1-labels)**gamma)*tf.log(1-logits)
    return tf.reduce_sum(fl_loss)
def my_loss(y_true,y_prob):
    return K.sum(K.abs(y_true-y_prob))

if __name__=="__main__":
    model, (train_x, train_x)=get_model()
    model.summary()