# Framework for image similarity program

import tkinter as tk
from hyperspy.api import load
from tkinter import filedialog
import numpy as np
from pixstem.api import PixelatedSTEM
from PIL import Image, ImageTk
from scipy.ndimage.filters import gaussian_filter
import requests

# global variables
file = None  # user selected file 
selected_points = []  # user selected diffraction patterns


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
def denoise_image(orig_image):
    denoise_radius = 3
    denoised_image = gaussian_filter(orig_image, denoise_radius)
    return denoised_image


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
            surface_img[i].append(int(np.round(np.mean(np.asarray(denoise_image(temp_array))))))
        if i != len(stem_file.data) - 1:
            surface_img.append([])

    surface_img_arr = np.asarray(surface_img)
    surface_img = Image.fromarray(np.asarray(surface_img), mode='L')
    surface_img.save('surface image.jpeg', format='jpeg')

    return surface_img_arr


# point selection interface
def start_analysis():
    global file, selected_points
    if file is not None:
        def reset_points():
            global selected_points
            selected_points = []
            analysis_log['text'] = "Similarity mapping: please click on up to three points you would like to " \
                                   "use to analyze the phase similarity.\n"
    
        def confirm_point(point):
            global selected_points
            selected_points.append(point)
    
        def get_mouse_xy(event):
            global selected_points, file
            nonlocal surface_img_arr
    
            length = len(surface_img_arr)
            point_num = len(selected_points) + 1
            point = (int(event.x * length / 400), int(event.y * length / 400))  # get the mouse position from event
    
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
            preview_img = np.asarray(file.data[int(event.y * length / 400)][int(event.x * length / 400)])
            preview_img = Image.fromarray(preview_img).resize((400, 400))
            preview_img = ImageTk.PhotoImage(image=preview_img)
            r.preview_img = preview_img
            c2.itemconfigure(point_img, image=preview_img)
    
            r.point = point
            confirm_button.configure(command=lambda: confirm_point(point))
            r.update()

        def finalize_points():
            global selected_points
            if len(selected_points) >= 2:
                print("Selected points (x,y):")
                for i in range(len(selected_points)):
                    print(f"point{i + 1} = {selected_points[i][0]}, {selected_points[i][1]}")
                analysis_log['text'] = analysis_log['text'] + "Starting analysis...\n"
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

        # canvas for surface image
        c1 = tk.Canvas(r, width=400, height=400)
        c1.place(relx=0.07, anchor='nw')
        surface_img_arr = create_surface_img(file)
        tk_image = Image.fromarray(surface_img_arr).resize((400, 400))
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
        analysis_log['text'] = "Similarity mapping: please click on up to three points you would like to " \
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
    
        r.mainloop()
        
    else:
        label3['text'] = "Please select a file and try again.\n"
        
        
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
    label3['text'] = "This program was originally designed by Aniket Patel and Aaron Barbosa \nand modified by " \
                     "Marcus Hansen and Ainiu Wang.\n"
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

    root.mainloop()
