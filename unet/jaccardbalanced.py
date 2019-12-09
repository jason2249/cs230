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

# load test data from files
print("Loading test data from files...")
test_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/balanced_data_test_256.npz')
test_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/unet/balanced_label_test_256.npz')

test_data = test_data_file["arr_0"]
test_labels = test_labels_file["arr_0"]
print("Test data loaded")

#model = load_model('jaccardbalancesavedmodel.h5', custom_objects={'jaccard_distance_loss': jaccard_distance_loss})
model = load_model('jaccardbalanced256model.h5', custom_objects={'jaccard_distance_loss':jaccard_distance_loss, 'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
#model.compile(optimizer=Adam(), loss=jaccard_distance_loss)
_,accuracy, f1, precision, recall  = model.evaluate(test_data, test_labels)
print('Accuracy: %.2f' % (accuracy*100))
print('f1: %.2f' % (f1))
print('precision: %.2f' % (precision))
print('recall: %.2f' % (recall))

predictions = model.predict(test_data)

print("saving predictions to file...")
np.save("balanced_jaccard_predictions_256", predictions)
print("finished saving predictions to file")
    

print(np.sum(test_labels))
print(np.sum(predictions))


