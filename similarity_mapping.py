import tkinter as tk
import PIL.ImageShow
from hyperspy.api import load
from tkinter import filedialog
import numpy as np
from os import remove, path
from pixstem.api import PixelatedSTEM
from PIL import Image, ImageTk
from scipy.ndimage.filters import gaussian_filter
import requests

file = None
analysis_points = []


def load_file():
    global file

    input_file = filedialog.askopenfilename()
    label1['text'] = label1['text'] + "Loading file...\n"
    root.update()
    try:
        file = PixelatedSTEM(load(input_file))
        label1['text'] = label1['text'] + "File loaded.\n"
    except:
        label1['text'] = label1['text'] + "Error loading. Please check path and try again.\n"
    entry.delete(0, tk.END)


def denoise_image(orig_image):
    denoise_radius = 3
    denoised_image = gaussian_filter(orig_image, denoise_radius)
    return denoised_image


def create_surface_image(stem_file):

    # creates equal sized # of sections to take the center of the image
    sections = 8
    image_length = len(stem_file.data[0][0])
    section_size = image_length / sections
    section1 = int((image_length / 2) - (section_size / 2))
    section2 = int((image_length / 2) + (section_size / 2))

    # Creates the surface image by slicing the center section of all images in the block file and averaging it for
    # their respective pixel value in the 4D array.
    surface_image = [[]]
    temp_array = None
    for i in range(len(stem_file.data)):
        for j in range(len(stem_file.data[i])):
            # creates a horizontal slice of the image
            temp_slice = np.asarray(stem_file.data[i][j][section1:section2])

            # refines to slice to be a square
            for r in range(len(temp_slice)):
                temp_array = temp_slice[r][section1:section2]

            # takes the average value of the pixels in the slice as adds them to an array that will be the surface image
            surface_image[i].append(int(np.round(np.mean(np.asarray(denoise_image(temp_array))))))
        if i != len(stem_file.data) - 1:
            surface_image.append([])

    surface_image_array = np.asarray(surface_image)
    surface_image = Image.fromarray(np.asarray(surface_image), mode='L')
    surface_image.save('surface image.jpeg', format='jpeg')

    return surface_image_array


def start_analysis(values=None):
    global file, analysis_points

    def reset_coords():
        global analysis_points
        analysis_points = []
        analysis_log['text'] = "Similarity mapping: please click on up to three points you would like to " \
                               "use to analyze the phase similarity.\n"

    def confirm_coords():
        global analysis_points

        if len(analysis_points) >= 2:
            print("Selected points (x,y):")
            for i in range(len(analysis_points)):
                print(f"point{i+1} = {analysis_points[i][0]}, {analysis_points[i][1]}")
            analysis_log['text'] = analysis_log['text'] + "Starting analysis...\n"
            # analysis(point1, values, point2)
            # remove("temp.png")
            # c2.unbind('<Button-1>')
            # r.destroy()
            # label1['text'] = label1['text'] + "Analysis complete.\n"
        else:
            reset_coords()

    def confirm_point(point):
        global analysis_points

        analysis_points.append(point)

    def mouse_coords(event):
        global analysis_points, file
        nonlocal surface_image

        length = len(surface_image)
        point_num = len(analysis_points) + 1
        point = (int(event.x * length / 400), int(event.y * length / 400))  # get the mouse position from event

        # deletes the last entry line if the point hasnt been confirmed to avoid log spam. refreshes the current point
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
        analysis_log['text'] = analysis_log['text'] + f"point{point_num} = " + str(point[0]) + " " + str(point[1])+"\n"
        print(f"point{point_num} ", point)

        tk_point_image = np.asarray(file.data[int(event.y * length / 400)][int(event.x * length / 400)])
        tk_point_image = Image.fromarray(tk_point_image).resize((400, 400))
        tk_point_image = ImageTk.PhotoImage(image=tk_point_image)
        analysis_window.tk_point_image = tk_point_image
        selected_point_canvas.itemconfigure(point_img, image=tk_point_image)

        analysis_window.point = point
        confirm_button.configure(command=lambda: confirm_point(point))
        analysis_window.update()

    surface_image = create_surface_image(file)

    analysis_window = tk.Toplevel(root)
    analysis_window.title('')

    canvas_height = 720
    canvas_width = 1280
    analysis_canvas = tk.Canvas(analysis_window, height=canvas_height, width=canvas_width)
    analysis_canvas.pack()

    analysis_frame = tk.Frame(analysis_window, bg='#FFFFFF')
    analysis_frame.place(relwidth=1, relheight=1)

    analysis_log = tk.Message(analysis_frame, bg='#FFFFFF', font=('Calibri', 15), anchor='nw', justify='left',
                              highlightthickness=0, bd=0, width=canvas_width * 0.9)
    analysis_log.place(relx=0.05, rely=0.65, relwidth=0.9, relheight=0.25)

    reset_button = tk.Button(analysis_frame, text='Reset', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                             bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                             command=lambda: reset_coords(), pady=0.02, fg='#373737', borderwidth='2',
                             relief="groove")
    reset_button.place(relx=0.15, rely=0.90, relwidth=0.20, relheight=0.07)

    confirm_button = tk.Button(analysis_frame, text='Confirm Point', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                               bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                               command=lambda: confirm_point(None), pady=0.02, fg='#373737', borderwidth='2',
                               relief="groove")
    confirm_button.place(relx=0.40, rely=0.90, relwidth=0.20, relheight=0.07)

    analyze_button = tk.Button(analysis_frame, text='Analyze points', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                               bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                               command=lambda: confirm_coords(), pady=0.02, fg='#373737', borderwidth='2',
                               relief="groove")
    analyze_button.place(relx=0.65, rely=0.90, relwidth=0.20, relheight=0.07)

    surface_image_canvas = tk.Canvas(analysis_window, width=400, height=400)
    surface_image_canvas.place(relx=0.1, anchor='nw')
    tk_image = Image.fromarray(surface_image).resize((400, 400))
    tk_image = ImageTk.PhotoImage(image=tk_image)
    surface_image_canvas.create_image(0, 0, anchor='nw', image=tk_image)

    selected_point_canvas = tk.Canvas(analysis_window, width=400, height=400)
    selected_point_canvas.place(relx=0.6, anchor='nw')
    point_img = selected_point_canvas.create_image(0, 0, anchor='nw', image=None)

    surface_image_canvas.bind('<Button-1>', mouse_coords)
    analysis_log['text'] = "Similarity mapping: please click on up to three points you would like to " \
                           "use to analyze the phase similarity.\n"

    analysis_window.mainloop()
    if path.exists("temp.png"):
        remove("temp.png")


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
    label = tk.Label(frame, text='PED Similarity Mapping', bg='#FFFFFF', font=('Times New Roman', 40), fg='#373737')
    label.place(relx=0.15, rely=0.1, relwidth=0.7, relheight=0.1)

    # Text Output box
    label1 = tk.Message(frame, bg='#F3F3F3', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                        bd=0, width=1500, fg='#373737', borderwidth=2, relief="groove")
    label1['text'] = "This program was originally designed by Aniket Patel and Aaron Barbosa \nand modified by " \
                     "Marcus Hansen and Ainiu Wang.\n"
    label1.place(relx=0.1, rely=0.54, relwidth=0.8, relheight=0.32)

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
    # 
    # button2 = tk.Button(frame, text='Create Bar Chart', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
    #                     activebackground='#D4D4D4', activeforeground='#252525',
    #                     command=lambda: bar_chart(), pady=0.02, fg='#373737', borderwidth='2',
    #                     relief="groove")
    # button2.place(relx=0.29, rely=0.34, relwidth=0.42, relheight=0.05)
    # 
    # button3 = tk.Button(frame, text='Create Strain Heat Map', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
    #                     activebackground='#D4D4D4', activeforeground='#252525',
    #                     command=lambda: set_curr_func('heat map'), pady=0.02, fg='#373737', borderwidth='2',
    #                     relief="groove")
    # button3.place(relx=0.29, rely=0.46, relwidth=0.42, relheight=0.05)
    # 
    # button4 = tk.Button(frame, text='Export Distance Data to .csv', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
    #                     bd=0, activebackground='#D4D4D4', activeforeground='#252525',
    #                     command=lambda: to_csv(input_filename), pady=0.02, fg='#373737', borderwidth='2',
    #                     relief="groove")
    # button4.place(relx=0.29, rely=0.40, relwidth=0.42, relheight=0.05)

    root.mainloop()
    if path.exists("temp.png"):
        remove("temp.png")
