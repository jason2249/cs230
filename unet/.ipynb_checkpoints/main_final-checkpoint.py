import keras.backend as K
from keras.models import load_model
from model import *
from data import *

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# load train data from files
print("Loading train data from files...")
# train_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_train.npy', allow_pickle=True)
# train_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_train.npy', allow_pickle=True)
train_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/full_train_data_rescale_256.npz')
train_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/binary_train_labels_256.npz')
train_data = train_data_file["arr_0"]
train_labels = train_labels_file["arr_0"]
print("Train data loaded")


model = unet()
model.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m, dice_coef])
model.fit(train_data, train_labels, epochs=1)
model.save('savedmodel_final_unet_256_binary_rescale.h5')
# # print train accuracy
# _, accuracy = model.evaluate(train_data, train_labels)
# print('Accuracy: %.2f' % (accuracy*100))

print("Loading test data from files...")
# test_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_test.npy', allow_pickle=True)
# test_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_test.npy', allow_pickle=True)
test_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/full_test_data_rescale_256.npz')
test_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/binary_test_labels_256.npz')

test_data = test_data_file["arr_0"]
test_labels = test_labels_file["arr_0"]
print("Test data loaded")

_, accuracy, f1, precision, recall, dice  = model.evaluate(test_data, test_labels)
print('Accuracy: %.2f' % (accuracy*100))
print('Accuracy: %.2f' % (accuracy*100))
print('f1: %.2f' % (f1))
print('precision: %.2f' % (precision))
print('recall: %.2f' % (recall))
print('dice: %.2f' % (dice))

predictions = model.predict(test_data)

print("saving predictions to file...")
np.savez("predictions_256_unet_rescale_binary.npz", predictions)
print("finished saving predictions to file")

# print test accuracy

