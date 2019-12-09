# import dependencies
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import scipy.misc
import imageio
from scipy import ndimage, misc

def load_data(filename_start):
    files = []
    home_dir = '/share/pi/hackhack/Breast/Breast_MRI/Breast_MRI_Annotated_n131'
    for directory in os.listdir(home_dir):
        print("outer loop")
        # for now, only use one directory
        if directory != "SEG_satoko":
            continue
        # do not include hidden files
        if directory[0] == ".":
            continue
        # iterate over patient directory files
        patient_dirs = os.listdir(home_dir + '/' + directory)
        for patient_dir in patient_dirs:
            if not patient_dir.isdigit():
                continue
            for file in os.listdir(home_dir + '/' + directory + '/' + patient_dir):
                # only consider files that start with filename_start
                if file[:len(filename_start)] == filename_start:
                    filename = home_dir + '/' + directory + '/' + patient_dir + '/' + file
                    example_filename = os.path.join(data_path, filename)
                    img = nib.load(example_filename)
                    data = img.get_fdata()
                    files.append(data)
                    break
    # convert array to numpy array
    print("finished processing need to make np array")
    return files

# makes all data of shape (512, 512)
def clean_data(dataset, tag):
    cleaned_data = []
    for image in dataset:
        print("processing image")
        image.shape
        if tag == "label":
            image = image[:,:,:,0]
        count = 0
        slice_dir = None
        for i, dim in enumerate(image.shape):
            if dim == 512:
                count += 1
            else:
                slice_dir = i
        if count != 2:
            continue
        if slice_dir != None:
            print("slicing image")
            if slice_dir == 0:
                for j in range(1, (image.shape[0] // 16)):
                    start = (j - 1) * 16
                    end = j * 16
                    new_img = image[start:end,:,:]
                    new_img = new_img.reshape((512, 512, 16))
                    new_img = ndimage.zoom(new_img, [0.5, 0.5, 1])
                    assert(new_img.shape == (256, 256, 16))
                    cleaned_data.append(new_img)
            if slice_dir == 1:
                for j in range(1, (image.shape[1] // 16)):
                    start = (j - 1) * 16
                    end = j * 16
                    new_img = image[:,start:end,:]
                    new_img = new_img.reshape((512, 512, 16))
                    new_img = ndimage.zoom(new_img, [0.5, 0.5, 1])
                    assert(new_img.shape == (256, 256, 16))
                    cleaned_data.append(new_img)
            if slice_dir == 2:
                for j in range(1, (image.shape[2] // 16)):
                    start = (j - 1) * 16
                    end = j * 16
                    new_img = image[:,:,start:end]
                    new_img = new_img.reshape((512, 512, 16))
                    new_img = ndimage.zoom(new_img, [0.5, 0.5, 1])
                    assert(new_img.shape == (256, 256, 16))
                    cleaned_data.append(new_img)
    
    return cleaned_data

# tag: label or data
# split: train, dev
def save_data(tag, split, array):
    numpyarray = np.array(array)
    # alternate
    # numpyarray = np.reshape(numpyarray.shape[0], 0, 512, 512, 0)
    # end alternate
    # original
    # numpyarray = np.expand_dims(numpyarray, axis=3)
    # end original
    np.savez("nvidia_{}_{}.npz".format(tag, split), numpyarray)

# load data and labels
def main():
    print("Start processing data...")

    dataset = load_data("volume")
    cleaned_dataset = clean_data(dataset, "data")
#     new_len = int(0.3 * len(cleaned_dataset))
#     cleaned_dataset = cleaned_dataset[:new_len]
#     split_cutoff = int(0.8 * len(cleaned_dataset))
#     train_data = cleaned_dataset[:split_cutoff]
#     test_data = cleaned_dataset[split_cutoff:]
#     save_data("data", "train", train_data)
#     save_data("data", "test", test_data)
    save_data("data", "full_256_16", cleaned_dataset)
    print("Saved data to file")

    labels = load_data("SEG")
    cleaned_labels = clean_data(labels, "label")
#     cleaned_labels = cleaned_labels[:new_len]
#     split_cutoff = int(0.8 * len(cleaned_labels))
#     train_labels = cleaned_labels[:split_cutoff]
#     test_labels = cleaned_labels[split_cutoff:]
#     save_data("label", "train", train_labels)
#     save_data("label", "test", test_labels)
    save_data("label", "full_256_16", cleaned_labels)
    print("Saved labels to file")

    assert(len(cleaned_labels) == len(cleaned_dataset))
    
if __name__ == '__main__':
    main()
