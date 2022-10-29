
import numpy as np
from skimage.io import imread, imshow, show
from skimage.transform import resize
import pickle


CATEGORTES = ['car png'  , 'door png' , 'ice cream png']

model = pickle.load(open('model.sav', 'rb'))


while True:
    path = input('Enter your URL: ')

    if path.lower() == 'stop':
        break

    try:
        img = imread(path)
    except:
        print("Imgae not found!")
        continue
    

    img_resize = resize(img,(64,64,3))
    flat_data = [img_resize.flatten()]
    flat_data = np.array(flat_data)

    y_out = model.predict(flat_data)
    y_out = CATEGORTES[y_out[0]]



    print(f'PREDICTED photo:{y_out}')

    imshow(img)
    show()


