from hyperspy.api import load
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pixstem.api import PixelatedSTEM
from PIL import Image, ImageTk, ImageOps, ImageFilter
from scipy.ndimage import gaussian_filter
from math import floor
import tensorflow as tf
from multiprocessing import Pool
import requests
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

file = None


# opens file explorer for user to select file
# assigns selected file to global file variable
def load_file():
    global file
    label_output['text'] = "Loading file...\n"
    input_file = filedialog.askopenfilename()
    r.update()
    try:
        file = PixelatedSTEM(load(input_file))
        label_output['text'] = label_output['text'] + "File loaded.\n"
    except ValueError:
        label_output['text'] = label_output['text'] + "Please select a file and try again.\n"
    except OSError:
        label_output['text'] = label_output['text'] + "Error loading. Please check the file path and try again.\n"


def start_analysis():
    global file

    label_output['text'] = "Starting analysis. This may take a while.\n"
    r.update()

    # generate feature vectors for all images in blockfile
    def extract_features(arr, model1):
        # load the image as 224x224
        arr = gaussian_filter(arr, 2*floor((int(round(len(arr)/64.0)))/2)+1)
        img = Image.fromarray(arr)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        # img = img.filter(ImageFilter.GaussianBlur)
        img = ImageOps.autocontrast(img, cutoff=1)
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
    for i in range(len(file.data)):
        for j in range(len(file.data[0])):
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

    # reduce the amount of dimensions in the feature vector
    pca = KernelPCA(n_components=100, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    ####################################################################################################################
    # get statistics for k-means clustering
    inertia_list = []
    max_k = 10
    k_range = list(range(1, max_k + 1))
    for k_val in k_range:
        # cluster feature vectors
        kmeans1 = KMeans(n_clusters=k_val, random_state=22)
        kmeans1.fit(x)
        inertia_list.append(kmeans1.inertia_)

    print("Inertia list: ", inertia_list)

    # automatic k detection using elbow method and distance calculation
    """distances = []
    for i in range(max_k):
        p1 = np.array([1, inertia_list[0]])  # line starting point
        p2 = np.array([max_k, inertia_list[max_k - 1]])  # line end point
        p3 = np.array([i + 1, inertia_list[i]])  # point compared against line
        d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
        distances.append(d)

    print("distances: ", distances)
    auto_k_value = distances.index(max(distances)) + 1
    print(str(auto_k_value))
    kmeans = KMeans(n_clusters=auto_k_value, random_state=22)
    kmeans.fit(x)"""

    # create new window showing elbow graph of k value vs. inertia
    fig, a = plt.subplots(figsize=(6, 5.5))
    x_val = k_range
    y_val = inertia_list

    plt.plot(x_val, y_val)
    plt.xlabel('K value')
    plt.ylabel('Inertia')
    plt.title('Elbow Graph')

    elbow_window = tk.Toplevel(r)
    elbow_window.geometry('600x600')
    chart_type = FigureCanvasTkAgg(plt.gcf(), elbow_window)
    chart_type.draw()
    chart_type.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1)

    ####################################################################################################################
    # analysis window
    r1 = tk.Toplevel(r)
    r1.title('')

    canvas_height = 640
    canvas_width = 1000
    c1 = tk.Canvas(r1, height=canvas_height, width=canvas_width)
    c1.pack()

    f1 = tk.Frame(r1, bg='#FFFFFF')
    f1.place(relwidth=1, relheight=1)

    # instructions for selecting k value for k-means clustering
    label3 = tk.Message(f1, bg='#FFFFFF', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                        bd=0, width=800, fg='#373737')
    label3['text'] = "Enter a k value for k-means clustering as a single integer.\n"
    label3.place(relx=0.07, rely=0.7, relwidth=0.86, relheight=0.1)

    heatmap_length = 0

    # update similarity map after new k value is entered
    def update_map(event):
        nonlocal heatmap_length
        k = int(entry.get())
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=22)
        kmeans.fit(x)

        heatmap_arr = []
        for filename, cluster in zip(filenames, kmeans.labels_):
            heatmap_arr.append(int(cluster) * 40)
        heatmap_arr_2d = np.reshape(heatmap_arr, (-1, len(file.data)))  # change according to data size
        heatmap_length = len(heatmap_arr_2d)
        print(heatmap_arr_2d)

        tk_image = Image.fromarray(heatmap_arr_2d).resize((400, 400))
        tk_image = ImageTk.PhotoImage(image=tk_image)
        r1.tk_image = tk_image
        c2.itemconfigure(temp_img, image=tk_image)

        r1.update()

    # entry box for k value
    entry = tk.Entry(f1, bg='#F3F3F3', font=('Calibri', 18), justify='left', highlightthickness=0,
                     bd=0, width=800, fg='#373737', borderwidth=2, relief="groove")
    entry.place(relx=0.4, rely=0.8, relwidth=0.2, relheight=0.08)
    entry.bind("<Return>", update_map)

    # display diffraction pattern in the blockfile selected via cursor
    def get_mouse_xy(event):
        nonlocal heatmap_length
        length = heatmap_length
        if heatmap_length != 0:
            point = (int(event.x * length / 400), int(event.y * length / 400))  # get the mouse position from event

            # displays selected diffraction pattern from .blo file
            preview_img = np.asarray(file.data[int(event.y * length / 400)][int(event.x * length / 400)])
            preview_img = gaussian_filter(preview_img, 2*floor((int(round(len(preview_img)/64.0)))/2)+1)
            preview_img = Image.fromarray(preview_img).resize((400, 400))
            # preview_img = preview_img.filter(ImageFilter.GaussianBlur)
            preview_img = ImageOps.autocontrast(preview_img, cutoff=1)
            preview_img = ImageTk.PhotoImage(image=preview_img)
            r1.preview_img = preview_img
            c3.itemconfigure(point_img, image=preview_img)

        r1.update()

    # canvas for surface image
    c2 = tk.Canvas(r1, width=400, height=400)
    c2.place(relx=0.07, anchor='nw')
    temp_img = c2.create_image(0, 0, anchor='nw', image=None)
    c2.bind('<Button-1>', get_mouse_xy)

    # preview diffraction pattern
    c3 = tk.Canvas(r1, width=400, height=400)
    c3.place(relx=0.93, anchor='ne')
    point_img = c3.create_image(0, 0, anchor='nw', image=None)

    r1.mainloop()


if __name__ == "__main__":
    HEIGHT = 700
    WIDTH = 800

    r = tk.Tk()
    r.title('')

    c = tk.Canvas(r, height=HEIGHT, width=WIDTH)
    c.pack()
    f = tk.Frame(r, bg='#FFFFFF')
    f.place(relwidth=1, relheight=1)

    # TAMU MSEN logo
    url = 'https://github.com/TAMU-Xie-Group/PED-Strain-Mapping/blob/main/msen.png?raw=true'
    logo = Image.open(requests.get(url, stream=True).raw)
    logo = logo.resize((200, 40))
    logo = ImageTk.PhotoImage(logo)
    label_logo = tk.Label(f, image=logo, bg='#FFFFFF')
    label_logo.place(relx=0.05, rely=0.05, anchor='w')

    # Menu Label
    label_title = tk.Label(f, text='Keras Similarity Mapping', bg='#FFFFFF', font=('Times New Roman', 40), fg='#373737')
    label_title.place(relx=0.15, rely=0.1, relwidth=0.7, relheight=0.1)

    # Text Output box
    label_output = tk.Message(f, bg='#F3F3F3', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                              bd=0, width=1500, fg='#373737', borderwidth=2, relief="groove")
    label_output['text'] = ""
    label_output.place(relx=0.1, rely=0.54, relwidth=0.8, relheight=0.32)

    # Buttons
    button_load = tk.Button(f, text='Load File', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                            activebackground='#D4D4D4', activeforeground='#252525',
                            command=lambda: load_file(), pady=0.02, fg='#373737', borderwidth='2',
                            relief="groove")
    button_load.place(relx=0.29, rely=0.22, relwidth=0.42, relheight=0.05)

    button_analyze = tk.Button(f, text='Start Analysis', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                               activebackground='#D4D4D4', activeforeground='#252525',
                               command=lambda: start_analysis(), pady=0.02, fg='#373737', borderwidth='2',
                               relief="groove")
    button_analyze.place(relx=0.29, rely=0.28, relwidth=0.42, relheight=0.05)

    r.mainloop()
