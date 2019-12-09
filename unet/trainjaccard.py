import keras.backend as K
from model import *
from data import *

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

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
train_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/balanced_data_train_256.npz', allow_pickle=True)
train_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/balanced_label_train_256.npz', allow_pickle=True)
train_data = train_data_file["arr_0"]
train_labels = train_labels_file["arr_0"]
print("Train data loaded")

model = unet()
model.compile(optimizer=Adam(lr=1e-4),loss=jaccard_distance_loss, metrics=['accuracy',f1_m,precision_m, recall_m])
model.fit(train_data, train_labels, epochs=5)
model.save('jaccardbalanced256model.h5')
# # print train accuracy
# _, accuracy = model.evaluate(train_data, train_labels)
# print('Accuracy: %.2f' % (accuracy*100))

# print test accuracy

