import numpy as np
from sklearn.model_selection import train_test_split

print("Loading data and labels...")
data_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/new_data_full_256.npz')
labels_file = np.load('/share/pi/hackhack/Breast/Breast_MRI/ericjason230/github_public/new_label_full_256.npz')
print("Loaded data and labels")

# print(labels.shape)
data = data_file["arr_0"]
print(data.shape)
labels = labels_file["arr_0"]
print(labels.shape)

train_X,valid_X,train_ground,valid_ground = train_test_split(data,
                                                             labels,
                                                             test_size=0.2,
                                                             random_state=13)

print("Save split...")
np.savez("full_train_data_256.npz", train_X)
np.savez("full_test_data_256.npz", valid_X)
np.savez("full_train_label_256.npz", train_ground)
np.savez("full_test_label_256.npz", valid_ground)
print("All data splits saved to file")