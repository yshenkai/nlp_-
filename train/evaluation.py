from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,Callback,EarlyStopping
from train.fine_get_model import get_model
import keras.backend as K
import numpy
from sklearn.metrics import f1_score, precision_score, recall_score
model,(train_x,train_y)=get_model()
def scheduler(epoch):
    if epoch% 100==0 and epoch != 0:
        lr=K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr,lr*0.5)
    return K.get_value(model.optimizer.lr)
reduce_lr_epoch=LearningRateScheduler(scheduler)
print("train_y",train_y.shape)
print("model.output",model.output.shape)

a=model.evaluate(train_x,train_y,batch_size=4)
print(a[0])
print(a[1])
print(a[2])

