# Framework for image similarity program
import sys
import os
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
import cv2
from math import sqrt
import scipy.spatial.distance as ssd


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
# def denoise(orig_image):
#     denoise_radius = 3
#     denoised_image = gaussian_filter(orig_image, denoise_radius)
#     denoised_image = Image.fromarray(denoised_image)
#     denoised_image = ImageOps.autocontrast(denoised_image, cutoff= )
#     denoised_image = np.array(denoised_image)
#     return denoised_image


# returns a representation of the .blo file as a 2d array that can be turned into an image
def create_surface_img(stem_file):
    # creates equal sized # of sections to take the center of the image
    sections = 8
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

    surface_img_arr = np.asarray(surface_img, dtype='uint8')
    # surface_img_arr = 255 - surface_img_arr
    surface_img = Image.fromarray(surface_img_arr, mode='L')
    surface_img = ImageOps.autocontrast(surface_img, cutoff=(1, 0))
    surface_img.save('surface image.jpeg', format='jpeg')
    surface_img_arr = np.asarray(surface_img)

    return surface_img_arr


# point selection interface
def start_analysis():
    global file, selected_points
    if file is not None:
        def reset_points():
            global selected_points
            nonlocal tk_img_arr, tk_image_orig, tk_img_arr_orig
            selected_points = []
            tk_img_arr = tk_img_arr_orig
            tk_img = ImageTk.PhotoImage(image=tk_image_orig)
            r.tk_img = tk_img
            c1.itemconfigure(bf_image, image=tk_img)
            r.update()
            analysis_log['text'] = "Similarity mapping: please click on points you would like to " \
                                   "use to analyze the phase similarity.\n"
    
        def confirm_point(point):
            global selected_points
            nonlocal surface_img_arr, tk_img_arr, tk_image
            analysis_log['text'] = analysis_log['text'][:-1] + ' - Confirmed\n'
            selected_points.append(point)
            tk_img_arr = np.array(tk_img_arr)
            tk_img_arr[point[1], point[0]] = [0, 255, 0]
            bf_x = len(tk_img_arr[0])
            bf_y = len(tk_img_arr)

            # adjusts the image size to scale up to 400 based on the aspect ratio of the surface image.
            if bf_x > bf_y:
                tk_img = Image.fromarray(tk_img_arr).resize((400, int((bf_y / bf_x) * 400)))
            elif bf_x < bf_y:
                tk_img = Image.fromarray(tk_img_arr).resize((int((bf_x / bf_y) * 400), 400))
            else:
                tk_img = Image.fromarray(tk_img_arr).resize((400, 400))
            tk_img = ImageTk.PhotoImage(image=tk_img)
            r.tk_img = tk_img
            c1.itemconfigure(bf_image, image=tk_img)
            r.update()
            # img = Image.fromarray(np.asarray(file.data[point[1]][point[0]]))
            # img.save(f'x{point[1]}_y{point[0]}.png')
    
        def get_mouse_xy(event):
            global selected_points, file
            nonlocal surface_img_arr, img_x, img_y, tk_img_arr, prev_point

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

            # colors a red dot for the unconfirmed point for clarity
            tk_img_arr = np.array(tk_img_arr)
            if prev_point:
                x = prev_point[0]
                y = prev_point[1]
                if (tk_img_arr[y][x] == [255, 0, 0]).all():
                    intensity = surface_img_arr[y][x]
                    tk_img_arr[y][x] = [intensity, intensity, intensity]

            tk_img_arr[point[1], point[0]] = [255, 0, 0]
            bf_x = len(tk_img_arr[0])
            bf_y = len(tk_img_arr)

            # adjusts the image size to scale up to 400 based on the aspect ratio of the surface image.
            if bf_x > bf_y:
                tk_img = Image.fromarray(tk_img_arr).resize((400, int((bf_y / bf_x) * 400)))
            elif bf_x < bf_y:
                tk_img = Image.fromarray(tk_img_arr).resize((int((bf_x / bf_y) * 400), 400))
            else:
                tk_img = Image.fromarray(tk_img_arr).resize((400, 400))
            tk_img = ImageTk.PhotoImage(image=tk_img)
            r.tk_img = tk_img
            c1.itemconfigure(bf_image, image=tk_img)
            prev_point = point

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

            # for testing
            # selected_points = [(39, 220), (64, 223), (126, 198), (191, 166), (270, 106), (74, 11)]  # SMA
            # selected_points = [(40, 52), (12, 15), (41, 14), (63, 3)]  # VO2
            # selected_points = [(182, 117), (186, 125), (71, 77), (173, 79), (189, 90)]  # Crazy SMA
            sim_type = sim_selected.get()
            if len(selected_points) >= 1:
                print("Selected points (x,y):")
                for i in range(len(selected_points)):
                    print(f"point{i + 1} = {selected_points[i][0]}, {selected_points[i][1]}")
                analysis_log['text'] = analysis_log['text'] + "Starting analysis...\n"
                analysis(selected_points, sim_type)
                c2.unbind('<Button-1>')
                r.destroy()
                label1['text'] = label1['text'] + "Analysis complete.\n"
            else:
                reset_points()

        def compare_points():
            global selected_points

            # selected_points = [(39, 220), (81, 219), (64, 223), (126, 198), (191, 166), (270, 106)]  # SMA sample
            # selected_points = [(40, 52), (80, 51), (12, 15), (41, 14), (63, 3)]  # VO2
            sim_type = sim_selected.get()
            if len(selected_points) >= 1:
                print("Selected points (x,y):")
                for i in range(len(selected_points)):
                    print(f"point{i + 1} = {selected_points[i][0]}, {selected_points[i][1]}")
                analysis_log['text'] = analysis_log['text'] + "Starting comparison...\n"
                sim_results = []
                base_img = file.data[selected_points[0][1]][selected_points[0][0]]
                for point in selected_points:
                    compare_img = file.data[point[1]][point[0]]
                    if sim_type == 'Euclidean':
                        sim_results.append(euclidean_similarity([base_img, compare_img]))
                    elif sim_type == 'SSIM':
                        sim_results.append(ssim_similarity([base_img, compare_img]))
                    elif sim_type == 'Cosine':
                        sim_results.append(cosine_similarity([base_img, compare_img]))
                    save_img = Image.fromarray(compare_img)
                    save_img.save(f'x{point[1]}_y{point[0]}.png')
                c2.unbind('<Button-1>')
                print(sim_type)
                for sim in sim_results:
                    print(f'{sim:.3f}, ', end='')
                print()
            else:
                reset_points()

            return

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
        prev_point = []
        img_x = len(surface_img_arr[0])
        img_y = len(surface_img_arr)
        tk_img_arr = np.zeros((img_y, img_x, 3), dtype='uint8')
        for row in range(len(surface_img_arr)):
            for col in range(len(surface_img_arr[row])):
                i = surface_img_arr[row][col]
                tk_img_arr[row][col] = [i, i, i]
        tk_img_arr_orig = tk_img_arr
        # adjusts the image size to scale up to 400 based on the aspect ratio of the surface image.
        if img_x > img_y:
            tk_image = Image.fromarray(surface_img_arr).resize((400, int((img_y/img_x) * 400)))
        elif img_x < img_y:
            tk_image = Image.fromarray(surface_img_arr).resize((int((img_x / img_y) * 400), 400))
        else:
            tk_image = Image.fromarray(surface_img_arr).resize((400, 400))
        tk_image_orig = tk_image
        # canvas for surface image

        if img_x > img_y:
            c1 = tk.Canvas(r, width=400, height=int((img_y/img_x) * 400))
        elif img_x < img_y:
            c1 = tk.Canvas(r, width=int((img_x / img_y) * 400), height=400)
        else:
            c1 = tk.Canvas(r, width=400, height=400)

        c1.place(relx=0.07, anchor='nw')
        tk_image = ImageTk.PhotoImage(image=tk_image)
        bf_image = c1.create_image(0, 0, anchor='nw', image=tk_image, tag='img')
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

        compare_point_button = tk.Button(f, text='Compare points', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                   bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                   command=lambda: compare_points(), pady=0.02, fg='#373737', borderwidth='2',
                                   relief="groove")
        compare_point_button.place(relx=0.65, rely=0.78, relwidth=0.20, relheight=0.07)

        sim_options = ['Euclidean', 'Cosine', 'SSIM']
        sim_selected = tk.StringVar()
        sim_selected.set(sim_options[0])
        sim_method = tk.OptionMenu(f, sim_selected, *sim_options)
        sim_method.place(relx=0.40, rely=0.78, relwidth=0.20, relheight=0.07)

        reset_points()
        r.mainloop()
        
    else:
        label3['text'] = "Please select a file and try again.\n"


def analysis(points, sim_type):
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
        pool = Pool(processes=12)
        if sim_type == 'Euclidean':
            for output in tqdm.tqdm(pool.imap(euclidean_similarity, processing_list),
                                    total=len(processing_list)):
                results.append(output)
                pass
        elif sim_type == 'SSIM':
            for output in tqdm.tqdm(pool.imap(ssim_similarity, processing_list), total=len(processing_list)):
                results.append(output)
                pass
        elif sim_type == 'Cosine':
            for output in tqdm.tqdm(pool.imap(cosine_similarity, processing_list),
                                    total=len(processing_list)):
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


# def mahalanobis_dist(img_arrays):
#     array1 = img_arrays[0].flatten()
#     array2 = img_arrays[1].flatten()
#     mag1 = np.linalg.norm(array1)
#     mag2 = np.linalg.norm(array2)
#     combined_array = np.vstack([array1, array2])
#     v = np.cov(combined_array.T)
#     inv_v = np.linalg.inv(v)
#     dist = ssd.mahalanobis(array1, array2, inv_v)
#     similarity = 1 - (dist / (abs(mag1) + abs(mag2)))  # normalizes the similarity from 0 (all different) to 1 (same)
#     return similarity


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


def blob_similarity(img_arrays, scale=1.0):
    def blob_detection(input_image, img_scale=1.0):
        input_image = 255 - input_image
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 1
        params.minRepeatability = 1
        # params.filterByColor = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1000
        # might be able to scale the min area to the center most dot based on the
        # input image magnification, scale, resolution

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(input_image)

        kps = []
        for kp in keypoints:
            kps.append([kp.pt[0], kp.pt[1], kp.size])

        # invert_input = 255 - input_image
        # im_with_keypoints = cv2.drawKeypoints(invert_input, keypoints, np.array([]), (0, 0, 255),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        # original_image = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # combined_img = cv2.addWeighted(resize_image(im_with_keypoints, (1 / img_scale)), 0.5, original_image, 0.5, 0)
        # cv2.imshow('Blob Keypoints', combined_img)
        # cv2.waitKey(0)
        return kps

    def resize_image(input_image, img_scale):
        width = int(input_image.shape[1] * img_scale)
        height = int(input_image.shape[0] * img_scale)
        dim = (width, height)
        scaled_image = cv2.resize(input_image, dim)
        return scaled_image

    array1 = img_arrays[0]
    array2 = img_arrays[1]
    w_count = 1
    w_pos = 4
    w_dia = 1
    kps1 = blob_detection(array1)
    kps2 = blob_detection(array2)

    # Count similarity
    if len(kps1) != 0:
        count_sim = len(kps2) / len(kps1)
        if count_sim > 1:
            count_sim = 1 / count_sim
    else:
        w_count = 0
        count_sim = 0

    # Position similarity
    pos_blob_sims = []
    closest_index = []
    for blob in kps1:
        min_dist = len(array1)
        for blob2 in kps2:
            dist = sqrt((blob[0] - blob2[0]) ** 2 + (blob[1] - blob2[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_index.append(kps2.index(blob2))
        point_sim = 1 - (min_dist / len(array1))
        pos_blob_sims.append(point_sim)
    pos_sim = np.average(pos_blob_sims)

    # Diameter similarity
    dia_blob_sims = []
    for i in range(len(kps1)):
        point_dia_sim = kps2[closest_index[i]][2] / kps1[i][2]
        if point_dia_sim > 1:
            point_dia_sim = 1 / point_dia_sim
        dia_blob_sims.append(point_dia_sim)
    dia_sim = np.average(dia_blob_sims)

    similarity = ((w_count * count_sim) + (w_pos * pos_sim) + (w_dia * dia_sim)) / (w_dia + w_count + w_pos)
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
    size = (similarity_values[0].shape[0], similarity_values[0].shape[1], 3)
    print(size)
    region_map_array = np.zeros(size, np.uint8)
    # region_colors_hsv = []
    # for i in range(points):
    #     if i % 2 == 0:
    #         region_colors_hsv.append((int(i * 255 / points), 255, 255))
    #     else:
    #         region_colors_hsv.append((int(i * 255 / points), 180, 180))
    region_colors_hsv = [(int(i * 255 / points), 255, 255) for i in range(points)]
    print(region_map_array)
    print(region_colors_hsv)
    bins = 100
    # for i in range(points):
    #     percentile = np.percentile(similarity_values[i], 95)
    #     std = np.std(similarity_values[i])
    #     print(percentile - (3*std/4))
    #     histogram = plt.hist(similarity_values[i].flat, bins=100)
    #     min_height = (similarity_values[i].shape[0] * similarity_values[i].shape[1]) / (bins * 50)
    #     print(similarity_values[i].shape[0] * similarity_values[i].shape[1])
    #     print(min_height)
    #     hist_peaks = find_peaks(histogram[0], height=min_height)
    #     print(histogram[0])
    #     print(histogram[1])
    #     print(hist_peaks)
    #     print(hist_peaks[0][-1])
    #     min_similarity = histogram[1][hist_peaks[0][-1]] * 0.95
    #     print(min_similarity)
    #     print(std)
    #     print()
    #     for y in range(len(similarity_values[i])):
    #         for x in range(len(similarity_values[i][y])):
    #             if similarity_values[i][y][x] >= min_similarity:
    #                 if region_map_array[y][x].all() != 0:
    #                     region_map_array[y][x][0] = (region_map_array[y][x][0] + region_colors_hsv[i][0]) / 2
    #                 else:
    #                     region_map_array[y][x] = region_colors_hsv[i]

    # cleans up closest similarity for any missing spaces, minimum similarity is 0.75
    for y in range(len(region_map_array)):
        for x in range(len(region_map_array[y])):
            if region_map_array[y][x].all() == 0:
                max_sim = 0.0
                min_sim = 0.0
                index = 0
                for i in range(len(similarity_values)):
                    if similarity_values[i][y][x] > max_sim:
                        max_sim = similarity_values[i][y][x]
                        index = i
                if max_sim >= min_sim:
                    region_map_array[y][x] = region_colors_hsv[index]

    region_map_image = Image.fromarray(region_map_array, mode='HSV')
    region_map_image = region_map_image.convert('RGB')
    region_map_image.show()
    region_map_image.save('region_map_hsv.png')

        
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
