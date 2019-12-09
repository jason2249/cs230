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


# load test data from files
print("Loading test data from files...")
test_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_test.npy', allow_pickle=True)
test_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_test.npy', allow_pickle=True)
print("Test data loaded")

model = load_model('dicesavedmodel.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
_, accuracy = model.evaluate(test_data, test_labels)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(test_data)


print("Printing predictions...")
for i in range(10):
    print(predictions[i])

print("Printing test labels...")
for i in range(10):
    print(test_labels[i])
    

print(np.sum(test_labels))
print(np.sum(predictions))


