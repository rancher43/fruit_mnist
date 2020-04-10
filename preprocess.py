from PIL import Image
import numpy as np
from os import listdir
from sklearn.preprocessing import OneHotEncoder
import pickle

train_images = []
train_labels = []
count = 1
for dirc in sorted(listdir('/home/ds/fruits')):
    for img_path in listdir('/home/ds/fruits/' + str(dirc)):
        try:
            im = Image.open('/home/ds/fruits/' + str(dirc) + "/" + str(img_path))
            im = im.resize((100, 100))
            train_images.append(np.array(im))
            train_labels.append(count)
        except:
            continue
    count += 1

train_labels = np.reshape(train_labels, (-1, 1))
enc = OneHotEncoder(categories='auto')
train_labels = enc.fit_transform(train_labels).toarray()
#print(train_labels[40])
#print(np.shape(train_images))
train_images = np.array(train_images) / 255.0

train_images_files = open('train_images', 'ab')
pickle.dump(train_images, train_images_files)
train_images_files.close()

train_labels_files = open('train_labels', 'ab')
pickle.dump(train_labels, train_labels_files)
train_labels_files.close()
