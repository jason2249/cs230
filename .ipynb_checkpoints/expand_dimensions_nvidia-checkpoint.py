import numpy as np

print("Loading data and labels...")
train_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/full_train_data_256_16.npz')
test_data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/full_test_data_256_16.npz')
train_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/full_train_label_256_16.npz')
test_labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/full_test_label_256_16.npz')
print("Loaded data and labels")

# print(labels.shape)
train_data = train_data_file["arr_0"]
print(train_data.shape)

train_data = np.expand_dims(train_data, axis=1)
train_data_cat = np.concatenate([train_data, train_data, train_data, train_data], axis=1)

print("Save train data...")
print(train_data_cat.shape)
np.savez("full_train_data_256_cat.npz", train_data_cat)

train_labels = train_labels_file["arr_0"]
print(train_labels.shape)

train_labels = np.expand_dims(train_labels, axis=1)
train_labels_cat = np.concatenate([train_labels, train_labels, train_labels], axis=1)

print("Save train label...")
print(train_labels_cat.shape)
np.savez("full_train_label_256_cat.npz", train_labels_cat)

test_data = test_data_file["arr_0"]
print(test_data.shape)

test_data = np.expand_dims(test_data, axis=1)
test_data_cat = np.concatenate([test_data, test_data, test_data, test_data], axis=1)

print("Save test data...")
print(test_data_cat.shape)
np.savez("full_test_data_256_cat.npz", test_data_cat)

test_labels = test_labels_file["arr_0"]
print(test_labels.shape)

test_labels = np.expand_dims(test_labels, axis=1)
test_labels_cat = np.concatenate([test_labels, test_labels, test_labels], axis=1)

print("Save test label...")
print(test_labels_cat.shape)
np.savez("full_test_label_256_cat.npz", test_labels_cat)


print("All data splits saved to file")