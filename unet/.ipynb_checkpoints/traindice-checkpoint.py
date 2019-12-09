import keras.backend as K
from model import *
from data import *

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# load train data from files
print("Loading train data from files...")
train_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_train.npy', allow_pickle=True)
train_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_train.npy', allow_pickle=True)
print("Train data loaded")


model = unet()
model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
model.fit(train_data, train_labels, epochs=1)
model.save('dicesavedmodel.h5')
# # print train accuracy
# _, accuracy = model.evaluate(train_data, train_labels)
# print('Accuracy: %.2f' % (accuracy*100))

# print test accuracy

