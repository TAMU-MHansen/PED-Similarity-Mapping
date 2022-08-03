# Framework for image similarity program
import sys
import tkinter as tk
from hyperspy.api import load
from tkinter import filedialog
import numpy as np
from numpy import arange
from pixstem.api import PixelatedSTEM
from PIL import Image, ImageTk, ImageOps, ImageFilter
from scipy.ndimage.filters import gaussian_filter
import requests
from multiprocessing import Pool
import tqdm
from scipy import spatial
from scipy.signal import argrelextrema, find_peaks
from pandas import DataFrame
import plotly.express as px
import SSIM_PIL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# global variables
file = None  # user selected file 
selected_points = []  # user selected diffraction patterns
similarity_values = None


# prompts file dialog for user to select file
def load_file():
    global file
    label3['text'] = "Loading file...\n"
    input_file = filedialog.askopenfilename()
    root.update()
    try:
        file = PixelatedSTEM(load(input_file))
        label3['text'] = label3['text'] + "File loaded.\n"
    except ValueError:
        label3['text'] = label3['text'] + "Please select a file and try again.\n"
    except OSError:
        label3['text'] = label3['text'] + "Error loading. Please check the file path and try again.\n"


# takes in an image and returns a filtered version (gaussian)
def gaussian_denoise(orig_image):
    denoise_radius = 3
    denoised_image = gaussian_filter(orig_image, denoise_radius)
    return denoised_image


# returns a representation of the .blo file as a 2d array that can be turned into an image
def create_surface_img(stem_file):
    # creates equal sized # of sections to take the center of the image
    sections = 6
    image_length = len(stem_file.data[0][0])
    section_size = image_length / sections
    section1 = int((image_length / 2) - (section_size / 2))
    section2 = int((image_length / 2) + (section_size / 2))

    # Creates the surface image by slicing the center section of all images in the block file and averaging it for
    # their respective pixel value in the 4D array.
    surface_img = [[]]
    temp_array = None
    for i in range(len(stem_file.data)):
        for j in range(len(stem_file.data[i])):
            # creates a horizontal slice of the image
            temp_slice = np.asarray(stem_file.data[i][j][section1:section2])

            # refines to slice to be a square
            for r in range(len(temp_slice)):
                temp_array = temp_slice[r][section1:section2]

            # takes the average value of the pixels in the slice as adds them to an array that will be the surface image
            surface_img[i].append(int(np.round(np.mean(np.asarray(temp_array)))))
        if i != len(stem_file.data) - 1:
            surface_img.append([])

    surface_img_arr = np.asarray(surface_img)
    surface_img = Image.fromarray(np.asarray(surface_img), mode='L')
    surface_img = ImageOps.autocontrast(surface_img, cutoff=1)
    surface_img.save('surface image.jpeg', format='jpeg')

    return surface_img_arr


# point selection interface
def start_analysis():
    global file, selected_points
    if file is not None:
        def reset_points():
            global selected_points
            selected_points = []
            analysis_log['text'] = "Similarity mapping: please click on points you would like to " \
                                   "use to analyze the phase similarity.\n"
    
        def confirm_point(point):
            global selected_points
            selected_points.append(point)
    
        def get_mouse_xy(event):
            global selected_points, file
            nonlocal surface_img_arr, img_x, img_y

            point_num = len(selected_points) + 1
            # point = (int(event.x * img_x / 400), int(event.y * img_y / 400))  # get the mouse position from event
            if img_x > img_y:
                point = (int(event.x * img_x / 400), int(event.y * img_x / 400))
            elif img_x < img_y:
                point = (int(event.x * img_y / 400), int(event.y * img_y / 400))
            else:
                point = (int(event.x * img_x / 400), int(event.y * img_y / 400))

            # deletes the last entry line if the point hasn't been confirmed and refreshes the current point
            split_log = analysis_log['text'].split('\n')
            log = ''
            for i in range(len(split_log)):
                temp = split_log[i].split(' ')
                if f'point{point_num}' not in temp:
                    log = log + split_log[i]
                    if i < len(split_log) - 1:
                        log = log + '\n'
            if log != '':
                analysis_log['text'] = log
            analysis_log['text'] = analysis_log['text'] + f"point{point_num} = " + str(point[0]) + " " + str(
                point[1]) + "\n"
            print(f"point{point_num} ", point)
    
            # displays selected diffraction pattern from .blo file
            preview_img = np.asarray(file.data[point[1]][point[0]])
            preview_img = Image.fromarray(preview_img).resize((400, 400))
            preview_img = ImageTk.PhotoImage(image=preview_img)
            r.preview_img = preview_img
            c2.itemconfigure(point_img, image=preview_img)
    
            r.point = point
            confirm_button.configure(command=lambda: confirm_point(point))
            r.update()

        def finalize_points():
            global selected_points
            if len(selected_points) >= 1:
                print("Selected points (x,y):")
                for i in range(len(selected_points)):
                    print(f"point{i + 1} = {selected_points[i][0]}, {selected_points[i][1]}")
                analysis_log['text'] = analysis_log['text'] + "Starting analysis...\n"
                analysis(selected_points)
                c2.unbind('<Button-1>')
                r.destroy()
                label1['text'] = label1['text'] + "Analysis complete.\n"
            else:
                reset_points()

        # main window
        r = tk.Toplevel(root)
        r.title('')
    
        canvas_height = 640
        canvas_width = 1000
        c = tk.Canvas(r, height=canvas_height, width=canvas_width)
        c.pack()
    
        f = tk.Frame(r, bg='#FFFFFF')
        f.place(relwidth=1, relheight=1)

        surface_img_arr = create_surface_img(file)
        img_x = len(surface_img_arr[0])
        img_y = len(surface_img_arr)
        # adjusts the image size to scale up to 400 based on the aspect ratio of the surface image.
        if img_x > img_y:
            tk_image = Image.fromarray(surface_img_arr).resize((400, int((img_y/img_x) * 400)))
        elif img_x < img_y:
            tk_image = Image.fromarray(surface_img_arr).resize((int((img_x / img_y) * 400), 400))
        else:
            tk_image = Image.fromarray(surface_img_arr).resize((400, 400))

        # canvas for surface image

        if img_x > img_y:
            c1 = tk.Canvas(r, width=400, height=int((img_y/img_x) * 400))
        elif img_x < img_y:
            c1 = tk.Canvas(r, width=int((img_x / img_y) * 400), height=400)
        else:
            c1 = tk.Canvas(r, width=400, height=400)

        c1.place(relx=0.07, anchor='nw')
        tk_image = ImageTk.PhotoImage(image=tk_image)
        c1.create_image(0, 0, anchor='nw', image=tk_image)
        c1.bind('<Button-1>', get_mouse_xy)
    
        # canvas for preview diffraction pattern
        c2 = tk.Canvas(r, width=400, height=400)
        c2.place(relx=0.93, anchor='ne')
        point_img = c2.create_image(0, 0, anchor='nw', image=None)
    
        # message log
        analysis_log = tk.Message(f, bg='#FFFFFF', font=('Calibri', 15), anchor='nw', justify='left',
                                  highlightthickness=0, bd=0, width=canvas_width * 0.9)
        analysis_log.place(relx=0.05, rely=0.65, relwidth=0.9, relheight=0.25)
        analysis_log['text'] = "Similarity mapping: please click on points you would like to " \
                               "use to analyze the phase similarity.\n"
    
        # interactive buttons
        reset_button = tk.Button(f, text='Reset', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                 bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                 command=lambda: reset_points(), pady=0.02, fg='#373737', borderwidth='2',
                                 relief="groove")
        reset_button.place(relx=0.15, rely=0.88, relwidth=0.20, relheight=0.07)
    
        confirm_button = tk.Button(f, text='Confirm Point', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                   bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                   command=lambda: confirm_point(None), pady=0.02, fg='#373737', borderwidth='2',
                                   relief="groove")
        confirm_button.place(relx=0.40, rely=0.88, relwidth=0.20, relheight=0.07)
    
        analyze_button = tk.Button(f, text='Analyze points', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                   bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                   command=lambda: finalize_points(), pady=0.02, fg='#373737', borderwidth='2',
                                   relief="groove")
        analyze_button.place(relx=0.65, rely=0.88, relwidth=0.20, relheight=0.07)
        reset_points()
        r.mainloop()
        
    else:
        label3['text'] = "Please select a file and try again.\n"


def analysis(points):
    global file, similarity_values

    # point = points[0]
    similarity_values = []
    for point in points:
        x_length = len(file.data[0])
        y_length = len(file.data)
        processing_list = [[]]
        i = 0
        for y in range(y_length):
            for x in range(x_length):
                processing_list.append([])
                processing_list[i].append(file.data[point[1]][point[0]])
                processing_list[i].append(file.data[y][x])
                i += 1

        del processing_list[-1]

        results = []
        pool = Pool(processes=None)
        for output in tqdm.tqdm(pool.imap_unordered(ssim_similarity, processing_list), total=len(processing_list)):
            results.append(output)
            pass
        pool.close()

        similarity = np.zeros((y_length, x_length))
        i = 0
        for y in range(y_length):
            for x in range(x_length):
                # if results[i] >= 0.99:
                #     similarity[y][x] = float('NaN')
                # else:
                similarity[y][x] = results[i]
                i += 1

        similarity_values.append(similarity)
    print(similarity_values)


def cosine_similarity(img_arrays):
    similarity = -1 * (spatial.distance.cosine(img_arrays[0].flatten(), img_arrays[1].flatten()) - 1)
    return similarity


def euclidean_similarity(img_arrays):
    array1 = img_arrays[0].flatten()
    array2 = img_arrays[1].flatten()
    mag1 = np.linalg.norm(array1)
    mag2 = np.linalg.norm(array2)
    dist = 0
    for n in range(len(array1)):
        dist += (int(array1[n]) - int(array2[n]))**2
    dist = np.sqrt(dist)
    similarity = 1 - (dist/(abs(mag1) + abs(mag2)))  # normalizes the similarity from 0 (all different) to 1 (same)
    return similarity


def ssim_similarity(img_arrays):
    array1 = Image.fromarray(img_arrays[0])
    array2 = Image.fromarray(img_arrays[1])
    similarity = SSIM_PIL.compare_ssim(array1, array2)
    return similarity


def heat_map():
    global similarity_values

    for point_values in similarity_values:
        df = DataFrame(point_values, columns=arange(len(point_values[0])), index=arange(len(point_values)))
        print(df)
        fig = px.imshow(df, color_continuous_scale='turbo')
        fig.show()


# creates a histogram pop-up UI
def create_histogram():
    global similarity_values

    if file is None:
        label1['text'] = "Please load a file before creating a histogram.\n"
    elif similarity_values is None:
        label1['text'] = "Please analyze the file before creating a histogram.\n"
    else:
        root.update()
        for p in range(len(similarity_values)):
            flattened_similarity = np.array(similarity_values[p]).flat
            for i in range(len(flattened_similarity)):
                if flattened_similarity[i] == 0:
                    flattened_similarity[i] = float('NaN')

            fig, a = plt.subplots(figsize=(6, 5.5))
            plt.xlabel('Distance from center peak', fontsize=10)
            plt.ylabel('Counts', fontsize=10)
            plt.title(f'Distance Counts: point{p+1}', fontsize=10)

            plt.hist(flattened_similarity, bins=50)

            bar_chart_window = tk.Toplevel(root)
            bar_chart_window.geometry('600x600')
            chart_type = FigureCanvasTkAgg(plt.gcf(), bar_chart_window)
            chart_type.draw()
            chart_type.get_tk_widget().place(relx=0.0, rely=0.0, relwidth=1)


def create_region_map():
    global similarity_values

    points = len(similarity_values)
    intensity_section = 255 / (points + 2)
    region_map_array = np.zeros(similarity_values[0].shape)
    print(np.zeros(similarity_values[0].shape))
    bins = 100
    for i in range(points):
        percentile = np.percentile(similarity_values[i], 95)
        std = np.std(similarity_values[i])
        print(percentile - (3*std/4))
        histogram = plt.hist(similarity_values[i].flat, bins=100)
        min_height = (similarity_values[i].shape[0] * similarity_values[i].shape[1]) / (bins * 50)
        print(similarity_values[i].shape[0] * similarity_values[i].shape[1])
        print(min_height)
        hist_peaks = find_peaks(histogram[0], height=min_height)
        print(histogram[0])
        print(histogram[1])
        print(hist_peaks)
        print(hist_peaks[0][-1])
        min_similarity = histogram[1][hist_peaks[0][-1]] * 0.95
        print(min_similarity)
        print(std)
        print()
        for y in range(len(similarity_values[i])):
            for x in range(len(similarity_values[i][y])):
                if similarity_values[i][y][x] >= min_similarity:
                    if region_map_array[y][x] != 0:
                        region_map_array[y][x] = int((region_map_array[y][x] + ((i + 1) * intensity_section)) / 2)
                    else:
                        region_map_array[y][x] = int((i + 1) * intensity_section)

    # cleans up closest similarity for any missing spaces, minimum similarity is 0.75
    for y in range(len(region_map_array)):
        for x in range(len(region_map_array)):
            if region_map_array[y][x] == 0:
                max_sim = 0
                index = 0
                for i in range(len(similarity_values)):
                    if similarity_values[i][y][x] > max_sim:
                        max_sim = similarity_values[i][y][x]
                        index = i
                if max_sim >= 0.75:
                    region_map_array[y][x] = int((index + 1) * intensity_section)

    region_map_image = Image.fromarray(region_map_array)
    region_map_image = region_map_image.convert('L')
    region_map_image.show()
    region_map_image.save('region_map.png')

        
if __name__ == "__main__":
    HEIGHT = 700
    WIDTH = 800

    root = tk.Tk()
    root.title('')

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
    canvas.pack()
    frame = tk.Frame(root, bg='#FFFFFF')
    frame.place(relwidth=1, relheight=1)

    # TAMU MSEN logo
    url = 'https://github.com/TAMU-Xie-Group/PED-Strain-Mapping/blob/main/msen.png?raw=true'
    msen_image = Image.open(requests.get(url, stream=True).raw)
    msen_image = msen_image.resize((200, 40))
    msen_image = ImageTk.PhotoImage(msen_image)
    label1 = tk.Label(frame, image=msen_image, bg='#FFFFFF')
    label1.place(relx=0.05, rely=0.05, anchor='w')

    # Menu Label
    label2 = tk.Label(frame, text='PED Similarity Mapping', bg='#FFFFFF', font=('Times New Roman', 40), fg='#373737')
    label2.place(relx=0.15, rely=0.1, relwidth=0.7, relheight=0.1)

    # Text Output box
    label3 = tk.Message(frame, bg='#F3F3F3', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                        bd=0, width=1500, fg='#373737', borderwidth=2, relief="groove")
    label3['text'] = "This program was designed by Marcus Hansen and Ainiu Wang.\n"
    label3.place(relx=0.1, rely=0.54, relwidth=0.8, relheight=0.32)

    # Entry box
    entry = tk.Entry(frame, bg='#F3F3F3', font=('Calibri', 15), justify='left', highlightthickness=0,
                     bd=0, width=1500, fg='#373737', borderwidth=2, relief="groove")
    entry.place(relx=0.1, rely=0.88, relwidth=0.8, relheight=0.05)

    # Buttons
    button = tk.Button(frame, text='Load File', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                       activebackground='#D4D4D4', activeforeground='#252525',
                       command=lambda: load_file(), pady=0.02, fg='#373737', borderwidth='2',
                       relief="groove")
    button.place(relx=0.29, rely=0.22, relwidth=0.42, relheight=0.05)

    button1 = tk.Button(frame, text='Start Analysis', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: start_analysis(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button1.place(relx=0.29, rely=0.28, relwidth=0.42, relheight=0.05)
    
    button2 = tk.Button(frame, text='Create Similarity Heat Map', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: heat_map(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button2.place(relx=0.29, rely=0.34, relwidth=0.42, relheight=0.05)

    button3 = tk.Button(frame, text='Create Similarity Histogram', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: create_histogram(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button3.place(relx=0.29, rely=0.40, relwidth=0.42, relheight=0.05)

    button4 = tk.Button(frame, text='Create Similarity Region Map', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: create_region_map(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button4.place(relx=0.29, rely=0.46, relwidth=0.42, relheight=0.05)

    root.mainloop()
