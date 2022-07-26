import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests
from pixstem.api import PixelatedSTEM
import hyperspy.api as hs
from hyperspy import io_plugins
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
import tqdm

file = None

# prompts file dialog for user to select file
def load_file():
    global file
    label3['text'] = "Loading file...\n"
    input_file = filedialog.askopenfilename()
    root.update()
    try:
        file = PixelatedSTEM(hs.load(input_file))
        label3['text'] = label3['text'] + "File loaded.\n"
    except ValueError:
        label3['text'] = label3['text'] + "Please select a file and try again.\n"
    except OSError:
        label3['text'] = label3['text'] + "Error loading. Please check the file path and try again.\n"


def save_block():
    global file

    file_array = file.data
    # x_length = len(file_array[0])
    # y_length = len(file_array)
    # img_length = len(file_array[0][0])

    results = []
    pool = Pool(processes=None)
    # runs the gaussian denoise function on all the images in the array
    for i in range(len(file_array)):
        for output in tqdm.tqdm(pool.imap_unordered(gaussian_denoise, file_array[i]), total=len(file_array[i])):
            results.append(output)
            pass
    pool.close()

    i = 0
    # reshapes the array back into the original shape from the flattened resutls
    for row in range(len(file_array)):
        for col in range(len(file_array[row])):
            file_array[row][col] = results[i]
            i += 1

    stem_file_array = hs.signals.Signal2D(file_array)
    io_plugins.blockfile.file_writer('FilteredBlock.blo', stem_file_array)


def gaussian_denoise(orig_image):
    denoise_radius = 3
    denoised_image = gaussian_filter(orig_image, denoise_radius)
    return denoised_image


if __name__ == '__main__':
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

    button1 = tk.Button(frame, text='Filter and Save', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: save_block(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button1.place(relx=0.29, rely=0.28, relwidth=0.42, relheight=0.05)


    root.mainloop()
