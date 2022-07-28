import tkinter as tk
from hyperspy.api import load
from tkinter import filedialog
import numpy as np
from pixstem.api import PixelatedSTEM
from PIL import Image, ImageTk
from scipy.ndimage.filters import gaussian_filter
import requests
from scipy.ndimage import gaussian_filter

# for loading/processing the images
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


if __name__ == "__main__":

    input_file = filedialog.askopenfilename()
    file = PixelatedSTEM(load(input_file))

    def extract_features(arr, model1):
        # load the image as 224x224
        img = Image.fromarray(arr)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1, 224, 224, 3)
        # prepare image for model
        img_x = preprocess_input(reshaped_img)
        # get the feature vector
        features = model1.predict(img_x, use_multiprocessing=True)
        return features

    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    data = {}

    # loop through each image in the dataset
    # for i in range(len(file.data)):
    #   for j in range(len(file.data[0])):
    for i in range(50):
        for j in range(50):
            # try to extract the features and update the dictionary
            image_from_arr = file.data[i][j]
            image_name = "img" + str(i) + "_" + str(j)
            print(image_name)
            feat = extract_features(image_from_arr, model)
            data[image_name] = feat

    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1, 4096)

    # get the unique labels (from the flower_labels.csv)
    unique_labels = (0, 1, 2, 3, 4)

    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=100, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    # cluster feature vectors
    kmeans = KMeans(n_clusters=len(unique_labels), random_state=22)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    """groups = {}
    for filename, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(filename)
        else:
            groups[cluster].append(filename)
            
        print(groups)"""

    heatmap_arr = []
    for filename, cluster in zip(filenames, kmeans.labels_):
        heatmap_arr.append(int(cluster)*40)
    heatmap_arr_2d = np.reshape(heatmap_arr, (-1, 50))
    print(heatmap_arr_2d)

    r = tk.Tk()
    r.title('')

    canvas_height = 640
    canvas_width = 1000
    c = tk.Canvas(r, height=canvas_height, width=canvas_width)
    c.pack()

    f = tk.Frame(r, bg='#FFFFFF')
    f.place(relwidth=1, relheight=1)

    def get_mouse_xy(event):
        length = len(heatmap_arr_2d)
        point = (int(event.x * length / 400), int(event.y * length / 400))  # get the mouse position from event

        # displays selected diffraction pattern from .blo file
        preview_img = np.asarray(file.data[int(event.y * length / 400)][int(event.x * length / 400)])
        preview_img = Image.fromarray(preview_img).resize((400, 400))
        preview_img = ImageTk.PhotoImage(master=r, image=preview_img)
        r.preview_img = preview_img
        c2.itemconfigure(point_img, image=preview_img)

        r.update()

    # canvas for surface image
    c1 = tk.Canvas(r, width=400, height=400)
    c1.place(relx=0.07, anchor='nw')
    tk_image = Image.fromarray(heatmap_arr_2d).resize((400, 400))
    tk_image = ImageTk.PhotoImage(master=r, image=tk_image)
    c1.create_image(0, 0, anchor='nw', image=tk_image)
    c1.bind('<Button-1>', get_mouse_xy)

    # preview diffraction pattern
    c2 = tk.Canvas(r, width=400, height=400)
    c2.place(relx=0.93, anchor='ne')
    point_img = c2.create_image(0, 0, anchor='nw', image=None)

    r.mainloop()
