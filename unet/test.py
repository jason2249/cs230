from keras.models import load_model
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

# load test data from files
print("Loading test data from files...")
# test_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_test.npy', allow_pickle=True)
# test_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_test.npy', allow_pickle=True)
test_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/full_test_data_rescale_256.npz')
test_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/full_test_label_rescale_256.npz')

test_data = test_data_file["arr_0"]
test_labels = test_labels_file["arr_0"]
print("Test data loaded")

model = load_model('savedmodel_256_rescale.h5', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
_, accuracy, f1, precision, recall  = model.evaluate(test_data, test_labels)
print('Accuracy: %.2f' % (accuracy*100))
print('Accuracy: %.2f' % (accuracy*100))
print('f1: %.2f' % (f1))
print('precision: %.2f' % (precision))
print('recall: %.2f' % (recall))

predictions = model.predict(test_data)

print("saving predictions to file...")
np.savez("predictions_256_unet_rescale.npz", predictions)
print("finished saving predictions to file")

# print("Printing predictions...")
# for i in range(10):
#     print(predictions[i])

# print("Printing test labels...")
# for i in range(10):
#     print(test_labels[i])
    

print(np.sum(test_labels))
print(np.sum(predictions))
