from model import *
from data import *

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
train_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/full_train_data_256.npz')
train_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/full_train_label_256.npz')

train_data = train_data_file["arr_0"]
train_labels = train_labels_file["arr_0"]
print("Train data loaded")


model = unet()
model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
model.fit(train_data, train_labels, epochs=5)
model.save('savedmodel_256_unet_epochs_5.h5')
# # print train accuracy
# _, accuracy = model.evaluate(train_data, train_labels)
# print('Accuracy: %.2f' % (accuracy*100))

# print test accuracy

