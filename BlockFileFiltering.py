import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
import requests
from pixstem.api import PixelatedSTEM
from hyperspy.api import load
import hyperspy.api as hs
from hyperspy import io_plugins
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from multiprocessing import Pool
import tqdm
from skimage.restoration import denoise_wavelet, denoise_bilateral, denoise_tv_bregman, denoise_tv_chambolle
from skimage import exposure
from skimage.morphology import reconstruction, ball, disk
from skimage.filters import rank
from cv2_rolling_ball import subtract_background_rolling_ball

# declaring global variables used between functions
file = None
selected_points = None
input_file_path = None


# prompts file dialog for user to select file
def load_file():
    global file, input_file_path
    label3['text'] = "Loading file...\n"
    input_file_path = filedialog.askopenfilename()
    root.update()

    # Loading file and error catching
    try:
        file = PixelatedSTEM(load(input_file_path))
        label3['text'] = label3['text'] + "File loaded.\n"
    except ValueError:
        label3['text'] = label3['text'] + "Please select a file and try again.\n"
    except OSError:
        label3['text'] = label3['text'] + "Error loading. Please check the file path and try again.\n"


# creates the virtual brightfield image to navigate the dataset
def create_brightfield_image(stem_file):
    # creates equal sized # of sections to take the center of the image
    sections = 2
    image_length = len(stem_file.data[0][0])
    section_size = image_length / sections
    section1 = int((image_length / 2) - (section_size / 2))
    section2 = int((image_length / 2) + (section_size / 2))

    # Creates the brightfield image by slicing the center section of all images in the block file and averaging it for
    # their respective pixel value in the 4D array.
    brightfield_img = [[]]
    temp_array = None
    for i in tqdm.tqdm(range(len(stem_file.data))):
        for j in range(len(stem_file.data[i])):
            # creates a horizontal slice of the image and applies auto contrast to create a more distinct image
            # Auto contrast significantly increases computational time
            temp_img = Image.fromarray(np.array(stem_file.data[i][j], dtype='uint8'))
            temp_img = ImageOps.autocontrast(temp_img, cutoff=1)

            temp_slice = np.array(temp_img)[section1:section2]

            # refines to slice to be a square
            for r in range(len(temp_slice)):
                temp_array = temp_slice[r][section1:section2]

            # takes the average value of the pixels in the slice as adds them to the brightfield image
            brightfield_img[i].append(int(np.round(np.mean(np.asarray(temp_array)))))
        if i != len(stem_file.data) - 1:
            brightfield_img.append([])
    brightfield_img = np.array(brightfield_img, dtype='uint8')
    brightfield_img = 255 - brightfield_img
    brightfield_img = Image.fromarray(brightfield_img)
    brightfield_img_arr = np.array(brightfield_img)
    brightfield_img.save('virtual brightfield image.jpeg', format='jpeg')
    return brightfield_img_arr


# function optimized for multiprocessing, given a data variable with an image and processing parameters in a list
def multiprocessing_filter(data):
    filter_type = data[1][0]
    radius = data[1][1]
    gamma = data[1][2]
    post_filter_type = data[1][4]
    post_radius = data[1][5]

    # checks if auto contrast is applied to the data and assigns variables as needed
    if data[1][3]:
        lower = data[1][3][0]
        upper = data[1][3][1]
    img_array = data[0]

    # processes the image data using the passed parameters
    img_array = filter_method(img_array, filter_type, radius)

    # post filter and gamma auto contrasting if the parameters exist
    if data[1][3]:
        pil_img = Image.fromarray(img_array)
        img_array = ImageOps.autocontrast(pil_img, cutoff=(lower, upper))
        img_array = np.array(img_array, dtype='uint8')

    img_array = filter_method(img_array, post_filter_type, post_radius)

    # post filter gamma correction
    if gamma != 1:
        img_array = gamma_correction(img_array, gamma)

    return img_array


# applies the correct filter method with the given parameters based on the passed denoise_method
def filter_method(image, denoise_method, radius):
    global file
    filtered_image = image
    if denoise_method == 'None':
        filtered_image = image
    elif denoise_method == 'Gaussian':
        radius = int(radius)
        filtered_image = gaussian_filter(image, radius)
    elif denoise_method == 'Non Local Means':
        radius = int(radius)
        filtered_image = cv2.fastNlMeansDenoising(image, h=radius)
    elif denoise_method == 'Rolling Ball Correction':
        filtered_image, background = np.array(subtract_background_rolling_ball(image, radius, light_background=False,
                                                                               use_paraboloid=False, do_presmooth=True))
    elif denoise_method == 'CLAHE':
        filtered_image = np.array(clahe_normalization(image, radius))
    elif denoise_method == 'Automatic CLAHE':
        filtered_image = np.array(automatic_clahe(image))
    elif denoise_method == 'Automatic Histogram Adjustment':
        filtered_image = np.array(automatic_hist_adjust(image))
    elif denoise_method == 'Regional Maxima':
        filtered_image = np.array(regional_maxima(image, radius), dtype='uint8')
    elif denoise_method == 'Adaptive Histogram Equalization':
        filtered_image = np.array(exposure.equalize_adapthist(image, clip_limit=radius)*255, dtype='uint8')
    elif denoise_method == 'Local Histogram Equalization':
        neighborhood = disk(radius)
        filtered_image = np.array(rank.equalize(image, footprint=neighborhood), dtype='uint8')
    elif denoise_method == 'Global Histogram Equalization':
        filtered_image = np.array(exposure.equalize_hist(image)*255, dtype='uint8')
    elif denoise_method == 'Centered Mask':
        filtered_image = centered_mask(image, radius)
    elif denoise_method == 'Normalize':
        filtered_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    elif denoise_method == 'Contrast Stretching':
        p2, pr = np.percentile(image, (2, radius))
        filtered_image = np.array(exposure.rescale_intensity(image, in_range=(p2, pr)), dtype='uint8')
    elif denoise_method == 'Median':
        radius = int(radius)
        filtered_image = medfilt(image, radius)
    return filtered_image


# def image_normalization(image, new_max):
#     img_min = np.min(image)
#     print(img_min)
#     img_max = np.max(image)
#     print(img_max)
#     new_image = np.zeros(image.shape)
#     for row in range(len(image)):
#         for col in range(len(image[row])):
#             if image[row][col] > 200:
#                 print(image[row][col])
#             new_image[row][col] = (image[row][col] - img_min) * (new_max / (img_max-img_min))
#     return new_image


def centered_mask(image, radius):
    center = (int(len(image[0])/2), int(len(image)/2))
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.circle(mask, center, int(radius), (255, 255, 255), -1)
    image = cv2.bitwise_and(image, mask)
    return image


def regional_maxima(image, h):
    seed = image - h
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    image = image - dilated
    return image


def clahe_normalization(image, radius):
    clahe = cv2.createCLAHE(clipLimit=radius, tileGridSize=(4, 4))
    image = clahe.apply(image)
    return image


# no user-input parameter
def automatic_clahe(image):
    avg_val = np.average(image)
    clip_lim = round(30.0 / avg_val) + 1
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(4, 4))
    image = clahe.apply(image)
    return image


# no user-input parameter
def automatic_hist_adjust(image):
    avg_val = np.average(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            val = image[i][j]
            val = val * (255 / (2 * avg_val + 40)) - avg_val
            if val > 255:
                val = 255
            if val < 0:
                val = 0
            image[i][j] = val
    return image

# # Work in Progress, doesn't do much.
# def custom_filter1(image):
#     avg_intensity = int(np.average(image))
#     intensity_multiplier = (0.5, 1.0)  # outside, inside
#     y_mid = int(len(image) / 2)
#     x_mid = int(len(image[y_mid]) / 2)
#     for y in range(len(image)):
#         for x in range(len(image[y])):
#             if image[y][x] < avg_intensity:
#                 image[y][x] = 0
#     return image


# # Decent on super noisy images, still a WIP
# def custom_filter2(image, radius):
#     check_radius = radius
#     for y in range(check_radius, len(image)-check_radius):
#         for x in range(check_radius, len(image[y])-check_radius):
#             radius_area = image[(y-check_radius):(y+check_radius+1), (x-check_radius):(x+check_radius+1)]
#             radius_area = radius_area.flatten()
#             np.delete(radius_area, int(len(radius_area)/2))
#             area_avg = np.average(radius_area)
#             if image[y][x] >= area_avg*1.25:
#                 image[y][x] = int(area_avg) * 0.5
#     return image


# Adjusts the given image based on the passed gamma value.
def gamma_correction(image, gamma):
    def adjust_gamma(img):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    # apply gamma correction and show the images
    adjusted = adjust_gamma(image)
    image = adjusted
    return image


# Uses OpenCV blob detection to try to find blobs in the diffraction pattern. min_area adjustable based on pattern size
def blob_detection(image, scale=1, min_area=50):
    # invert image for better detection
    image = 255 - image
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 1
    params.minRepeatability = 1
    # params.filterByColor = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = min_area
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
    keypoints = detector.detect(image)
    # Invert image for better detection
    invert_input = 255 - image
    # Draws the detected blob circles onto the image
    im_with_keypoints = cv2.drawKeypoints(invert_input, keypoints, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    combined_img = np.array(im_with_keypoints)
    return combined_img


# Main Analysis window
def start_analysis():
    global file, selected_points
    label3['text'] = label3['text'] + "Generating virtual bright-field image, this may take a few seconds\n"
    frame.update()
    if file is not None:
        # Previews the diffraction pattern at the selected point using the given parameters
        def preview_point(point):
            # assigning user parameters to variables
            print(point)
            radius = float(radius_value.get('1.0', 'end-1c'))
            filter_type = filter_clicked.get()
            gamma = float(gamma_value.get('1.0', 'end-1c'))
            post_radius = float(post_radius_value.get('1.0', 'end-1c'))
            post_filter_type = post_filter_clicked.get()

            # copies the point data to avoid altering the main data
            img_point = file.data[point[1]][point[0]].copy()
            filter_preview_img = filter_method(img_point, filter_type, radius)

            # auto contrast processing if checkbox enabled
            if cb_var.get():
                lower = int(cv_lower.get('1.0', 'end-1c'))
                upper = int(cv_upper.get('1.0', 'end-1c'))
                pil_img = Image.fromarray(filter_preview_img)
                filter_preview_img = ImageOps.autocontrast(pil_img, cutoff=(lower, upper))
                filter_preview_img = np.array(filter_preview_img)

            # secondary post filtering
            filter_preview_img = filter_method(filter_preview_img, post_filter_type, post_radius)

            # gamma processing
            if gamma != 1.0:
                filter_preview_img = gamma_correction(filter_preview_img, gamma)

            # blob detection if checkbox enabled
            if cp_blob.get():
                min_area = int(min_area_value.get('1.0', 'end-1c'))
                filter_preview_img = blob_detection(filter_preview_img, min_area=min_area)
                filter_preview_img = Image.fromarray(filter_preview_img, mode='RGB')

            # If the data is an array, converts it to a PIL Image due to potential conflicts with blob RGB and Grayscale
            if isinstance(filter_preview_img, np.ndarray):
                filter_preview_img = Image.fromarray(filter_preview_img)

            # resizes the image to fit the canvas and replace the current image with an updated filtered image
            filter_preview_img = filter_preview_img.resize((400, 400))
            filter_preview_img = ImageTk.PhotoImage(image=filter_preview_img)
            r.filter_preview_img = filter_preview_img
            filtered_canvas.itemconfigure(filtered_img_preview, image=filter_preview_img)

        def get_mouse_xy(event):
            global selected_points, file
            nonlocal brightfield_img_arr, img_x, img_y

            # get the mouse click position depending on the image shape due to resize scaling in rectangular images
            if img_x > img_y:
                point = (int(event.x * img_x / 400), int(event.y * img_x / 400))
            elif img_x < img_y:
                point = (int(event.x * img_y / 400), int(event.y * img_y / 400))
            else:
                point = (int(event.x * img_x / 400), int(event.y * img_y / 400))

            # displays selected diffraction pattern from .blo file
            preview_img = np.asarray(file.data[point[1]][point[0]])
            preview_img = Image.fromarray(preview_img).resize((400, 400))
            preview_img = ImageTk.PhotoImage(image=preview_img)
            r.preview_img = preview_img
            c2.itemconfigure(point_img, image=preview_img)

            r.point = point
            confirm_button.configure(command=lambda: preview_point(point))
            r.update()

        # processes the entire dataset for filtering with the given user parameters
        def filter_file():
            global file, input_file_path
            # assigns user parameters to variables
            radius = float(radius_value.get('1.0', 'end-1c'))
            filter_type = filter_clicked.get()
            gamma = float(gamma_value.get('1.0', 'end-1c'))
            post_radius = float(post_radius_value.get('1.0', 'end-1c'))
            post_filter_type = post_filter_clicked.get()

            if cb_var.get():
                lower = int(cv_lower.get('1.0', 'end-1c'))
                upper = int(cv_upper.get('1.0', 'end-1c'))
            else:
                lower = 0
                upper = 0
            file_array = np.zeros(np.array(file.data).shape, dtype='uint8')
            # creates a list for multiprocessing to pass each process the image and filter parameters
            multiprocessing_list = [[]]
            index = 0
            for y in range(len(file.data)):
                for x in range(len(file.data[y])):
                    multiprocessing_list.append([])
                    multiprocessing_list[index].append(file.data[y][x])
                    multiprocessing_list[index].append([filter_type, radius, gamma, (lower, upper), post_filter_type,
                                                        post_radius])
                    index += 1
            del multiprocessing_list[-1]
            results = []
            pool = Pool(processes=None)
            # runs the desired filtering method on all the images in the array
            # Processes fast but uses a lot of memory, can remove multiprocessing for reduced memory usage at the
            # cost of speed
            for output in tqdm.tqdm(pool.imap_unordered(multiprocessing_filter, multiprocessing_list),
                                    total=len(multiprocessing_list)):
                results.append(output)
                pass
            pool.close()

            i = 0
            # reshapes the array back into the original shape from the flattened results
            for row in range(len(file_array)):
                for col in range(len(file_array[row])):
                    file_array[row][col] = results[i]
                    i += 1
            stem_file_array = hs.signals.Signal2D(file_array)
            # saves the file with the original name plus suffixes based on the user parameters
            file_name = input_file_path[:-4]
            if filter_type != 'None':
                file_name = file_name + f'_{filter_type}_r{radius}'
            if gamma != 1:
                file_name = file_name + f'_g{gamma}'
            if lower & upper != 0:
                file_name = file_name + f'_ac({lower},{upper})'
            if post_filter_type != 'None':
                file_name = file_name + f'_{post_filter_type}_r{post_radius}'
            io_plugins.blockfile.file_writer(file_name + '.blo', stem_file_array)
            label3['text'] = label3['text'] + "Filtered file saved.\n"
            return

        # main window
        r = tk.Toplevel(root)
        r.title('')

        canvas_height = 800
        canvas_width = 1500
        c = tk.Canvas(r, height=canvas_height, width=canvas_width)
        c.pack()

        f = tk.Frame(r, bg='#FFFFFF')
        f.place(relwidth=1, relheight=1)

        brightfield_img_arr = create_brightfield_image(file)
        img_x = len(brightfield_img_arr[0])
        img_y = len(brightfield_img_arr)
        # adjusts the image size to scale up to 400 based on the aspect ratio of the virtual bright-field image.
        if img_x > img_y:
            tk_image = Image.fromarray(brightfield_img_arr).resize((400, int((img_y / img_x) * 400)))
        elif img_x < img_y:
            tk_image = Image.fromarray(brightfield_img_arr).resize((int((img_x / img_y) * 400), 400))
        else:
            tk_image = Image.fromarray(brightfield_img_arr).resize((400, 400))

        # canvas for the virtual brightfield image

        if img_x > img_y:
            c1 = tk.Canvas(r, width=400, height=int((img_y / img_x) * 400))
        elif img_x < img_y:
            c1 = tk.Canvas(r, width=int((img_x / img_y) * 400), height=400)
        else:
            c1 = tk.Canvas(r, width=400, height=400)

        c1.place(relx=0.05, anchor='nw')
        tk_image = ImageTk.PhotoImage(image=tk_image)
        c1.create_image(0, 0, anchor='nw', image=tk_image)
        c1.bind('<Button-1>', get_mouse_xy)

        # canvas for preview diffraction pattern
        c2 = tk.Canvas(r, width=400, height=400)
        c2.place(relx=0.5, anchor='n')
        point_img = c2.create_image(0, 0, anchor='nw', image=None)

        filtered_canvas = tk.Canvas(r, width=400, height=400)
        filtered_canvas.place(relx=0.95, anchor='ne')
        filtered_img_preview = filtered_canvas.create_image(0, 0, anchor='nw', image=None)

        # Image texts
        brightfield_text = tk.Label(f, text='Virtual Bright-Field Image', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        brightfield_text.place(relx=0.09, rely=0.50)

        orig_text = tk.Label(f, text='Original Image', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        orig_text.place(relx=0.45, rely=0.50)

        filtered_text = tk.Label(f, text='Filtered Image', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        filtered_text.place(relx=0.77, rely=0.50)

        # interactive buttons
        confirm_button = tk.Button(f, text='Preview', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                   bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                   command=lambda: preview_point(None), pady=0.02, fg='#373737', borderwidth='2',
                                   relief="groove")
        confirm_button.place(relx=0.40, rely=0.88, relwidth=0.20, relheight=0.07)

        # Filter File button
        analyze_button = tk.Button(f, text='Filter File', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0,
                                   bd=0, activebackground='#D4D4D4', activeforeground='#252525',
                                   command=lambda: filter_file(), pady=0.02, fg='#373737', borderwidth='2',
                                   relief="groove")
        analyze_button.place(relx=0.65, rely=0.88, relwidth=0.20, relheight=0.07)

        # filtering parameters buttons, labels, and text boxes
        filter_label = tk.Label(f, text='Filtering Method', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        filter_label.place(relx=0.05, rely=0.60)
        filter_options = ['None', 'Gaussian', 'Non Local Means', 'Rolling Ball Correction', 'CLAHE',
                          'Automatic CLAHE', 'Automatic Histogram Adjustment', 'Adaptive Histogram Equalization',
                          'Regional Maxima', 'Local Histogram Equalization', 'Global Histogram Equalization',
                          'Centered Mask', 'Normalize', 'Median', 'Contrast Stretching']
        filter_clicked = tk.StringVar()
        filter_clicked.set(filter_options[0])
        filter_dropdown = tk.OptionMenu(f, filter_clicked, *filter_options)
        filter_dropdown.place(relx=0.18, rely=0.60, relwidth=0.14, relheight=0.05)

        radius_text = tk.Label(f, text='Filtering Value', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        radius_text.place(relx=0.05, rely=0.67)
        radius_value = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        radius_value.place(relx=0.18, rely=0.67, relwidth=0.14, relheight=0.05)
        radius_value.insert('1.0', '3')

        gamma_label = tk.Label(f, text='Gamma Value', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        gamma_label.place(relx=0.05, rely=0.74)
        gamma_value = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        gamma_value.place(relx=0.18, rely=0.74, relwidth=0.14, relheight=0.05)
        gamma_value.insert('1.0', '1.0')

        cb_var = tk.BooleanVar()
        contrast_cb = tk.Checkbutton(f, text='Auto contrast?', bg='#FFFFFF', fg='#373737', font=('Calibri', 20),
                                     variable=cb_var, onvalue=True, offvalue=False)
        contrast_cb.place(relx=0.38, rely=0.60, relwidth=0.15, relheight=0.05)
        contrast_label = tk.Label(f, text='Contrast Cutoffs (lower, upper)', bg='#FFFFFF', font=('Calibri', 20),
                                  fg='#373737')
        contrast_label.place(relx=0.55, rely=0.60)
        cv_lower = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        cv_lower.place(relx=0.80, rely=0.60, relwidth=0.05, relheight=0.05)
        cv_lower.insert('1.0', '1')
        cv_upper = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        cv_upper.place(relx=0.86, rely=0.60, relwidth=0.05, relheight=0.05)
        cv_upper.insert('1.0', '1')

        cp_blob = tk.BooleanVar()
        blob_cb = tk.Checkbutton(f, text='Detect Blobs?', bg='#FFFFFF', fg='#373737', font=('Calibri', 20),
                                 variable=cp_blob, onvalue=True, offvalue=False)
        blob_cb.place(relx=0.38, rely=0.65, relwidth=0.15, relheight=0.05)
        min_area_label = tk.Label(f, text='Blob min. area', bg='#FFFFFF', font=('Calibri', 20),
                                  fg='#373737')
        min_area_label.place(relx=0.55, rely=0.65)
        min_area_value = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        min_area_value.place(relx=0.70, rely=0.65, relwidth=0.05, relheight=0.05)
        min_area_value.insert('1.0', '50')

        post_filter_label = tk.Label(f, text='Post Filtering Method', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        post_filter_label.place(relx=0.38, rely=0.70)
        post_filter_clicked = tk.StringVar()
        post_filter_clicked.set(filter_options[0])
        post_filter_dropdown = tk.OptionMenu(f, post_filter_clicked, *filter_options)
        post_filter_dropdown.place(relx=0.60, rely=0.70, relwidth=0.14, relheight=0.05)

        post_radius_text = tk.Label(f, text='Post Filtering Value', bg='#FFFFFF', font=('Calibri', 20), fg='#373737')
        post_radius_text.place(relx=0.38, rely=0.75)
        post_radius_value = tk.Text(f, bg='#FFFFFF', font=('Calibri', 20), fg='#373737', wrap='none')
        post_radius_value.place(relx=0.60, rely=0.75, relwidth=0.14, relheight=0.05)
        post_radius_value.insert('1.0', '3')

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
    try:
        url = 'https://github.com/TAMU-Xie-Group/PED-Strain-Mapping/blob/main/msen.png?raw=true'
        msen_image = Image.open(requests.get(url, stream=True).raw)
        msen_image = msen_image.resize((200, 40))
        msen_image = ImageTk.PhotoImage(msen_image)
        label1 = tk.Label(frame, image=msen_image, bg='#FFFFFF')
        label1.place(relx=0.05, rely=0.05, anchor='w')
    except:
        print('Error: no internet connection for TAMU MSEN logo')

    # Menu Label
    label2 = tk.Label(frame, text='Block File Filtering', bg='#FFFFFF', font=('Times New Roman', 40), fg='#373737')
    label2.place(relx=0.15, rely=0.1, relwidth=0.7, relheight=0.1)

    # Text Output box
    label3 = tk.Message(frame, bg='#F3F3F3', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness=0,
                        bd=0, width=1500, fg='#373737', borderwidth=2, relief="groove")
    label3['text'] = "This program was designed by Marcus Hansen.\n"
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

    button1 = tk.Button(frame, text='Filtering Preview', bg='#F3F3F3', font=('Calibri', 20), highlightthickness=0, bd=0,
                        activebackground='#D4D4D4', activeforeground='#252525',
                        command=lambda: start_analysis(), pady=0.02, fg='#373737', borderwidth='2',
                        relief="groove")
    button1.place(relx=0.29, rely=0.28, relwidth=0.42, relheight=0.05)

    root.mainloop()
