from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,Callback,EarlyStopping
from train.fine_get_model import get_model
import keras.backend as K
import numpy
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
metrics = Metrics()

reduce_lr_plat=ReduceLROnPlateau(monitor="loss",factor=0.1,patience=100,epsilon=0.0001,cooldown=0,min_lr=0)
early_stop=EarlyStopping(monitor="val_loss",patience=0)
checkpoint=ModelCheckpoint("../data_constract_xunlian/model_output/chandi/chandi_line_wise_recall_reduce_lr_epoch_weights.{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=1)
model,(train_x,train_y)=get_model()
def scheduler(epoch):
    if epoch% 100==0 and epoch != 0:
        lr=K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr,lr*0.5)
    return K.get_value(model.optimizer.lr)
reduce_lr_epoch=LearningRateScheduler(scheduler)
print("train_y",train_y.shape)
print("model.output",model.output.shape)


history=model.fit(train_x,train_y,batch_size=4,epochs=20,validation_split=0.1,shuffle=True,callbacks=[checkpoint,reduce_lr_epoch],verbose=1)
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history["myprecision_sta"])
plt.plot(history.history["myrecall_sta"])
plt.title("p-r")
plt.xlabel("epoch")
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("loss")
plt.xlabel("epoch")
plt.show()
