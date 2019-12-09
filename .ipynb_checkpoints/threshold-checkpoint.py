import numpy as np

# load from files
d = np.load('cleaned_data_train.npy', allow_pickle=True)
print("Loaded arrays half")
print(np.sum(d))
l = np.load('cleaned_label_train.npy', allow_pickle=True)
print("Loaded arrays")
print(np.sum(l))

# try thresholding once
output = d < 1
for i in range(100):
    for j in range(100):
        print(l[0][i][j])
for i in range(100):
    for j in range(100):
        print(output[0][i][j])

# try different values of thresholding
best_accuracy = None
best_threshold = None
for i in range(10):
    threshold = np.random.randint(100, 250)
    output = d > threshold
    num_correct = np.sum(np.equal(output, l))
    m, x, y, _ = output.shape
    total = m * x * y
    accuracy = num_correct / total
    print("Accuracy: ", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        
print("Best accuracy with %d with threshold %d" % (best_accuracy, best_threshold))