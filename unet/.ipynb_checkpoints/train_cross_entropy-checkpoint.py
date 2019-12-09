from model import *
from data import *

def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss



# load train data from files
print("Loading train data from files...")
train_data = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_data_train.npy', allow_pickle=True)
train_labels = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/cleaned_label_train.npy', allow_pickle=True)
print("Train data loaded")

model = unet()
model.compile(optimizer='adam', loss=weighted_cross_entropy, metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=1)
model.save('savedmodel_with_compile.h5')
# # print train accuracy
# _, accuracy = model.evaluate(train_data, train_labels)
# print('Accuracy: %.2f' % (accuracy*100))

# print test accuracy

