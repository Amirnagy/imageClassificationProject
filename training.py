
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
import pandas as pd

# support vector machine
import pickle
target = []
# my image
images = []
# data after transform 2d metrics to 1d metrics for every photo
flatten_data = []
directory = 'images'
CATEGORTES = ['car png', 'door png', 'ice cream png']

for category in CATEGORTES:
    class_num = CATEGORTES.index(category)
    path = os.path.join(directory, category)

    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))

        # normalize value from 0 to 1
        img_resize = resize(img_array, (64, 64, 3))

        # list in python
        flatten_data.append(img_resize.flatten())
        images.append(img_resize)
        target.append(class_num)

# import numpy as np
flatten_data = np.array(flatten_data)
target = np.array(target)
images = np.array(images)

# import matplotlib.pyplot as plt

unique, count = np.unique(target, return_counts=True)
fig = plt.figure(figsize = (10, 5))
plt.bar(CATEGORTES, count)
plt.show()


# split data to traning and testing
x_train, x_test, y_train, y_test = train_test_split(
    flatten_data, target, test_size=0.3, random_state=109)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


classifier = SVC(C=1.0, kernel='poly', random_state=0)
classifier.fit(x_train, y_train)

y_out = classifier.predict(x_test)
print(y_out)
'''
accuracy of classifier it take y_test of data and the the predicted data 
'''
accuracy_of_classifier = accuracy_score(y_test, y_out)
print ("Accuracy : ", accuracy_of_classifier)

'''
using pandas to list the real value and predicted values 
in the data frame  
'''
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_out})
print(df.to_string())

# save the model on the disk by pickle
filename = 'model.sav'
pickle.dump(classifier, open(filename, 'wb'))