# This is the main runner function from which the code is started
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import operator
from keras.models import model_from_json


def load_data_from_folder(folder_location="."):
    # indicates that directory path is being given
    # This for now will be a prototype function to load a single image
    images_data = []
    data_files = []
    for filename in os.listdir(folder_location):
        if filename.endswith(".jpg"):
            print("Working with {}.".format(filename))
            images_data.append(cv2.imread(os.path.join(folder_location, filename), cv2.IMREAD_GRAYSCALE))
            # cv2.imshow("t", images_data[0])
        elif filename.endswith(".dat"):
            with open(filename) as f:
                cleaned_data = [i.strip().split(" ") for i in f.readlines()[2:]]
                data_files.append(cleaned_data)
        else:
            continue
    return images_data, data_files


def recognize_board(image, threshhold_low=30, threshhold_high=60):
    # Other edge detection 1. Gauss + Sobel/Scharr 2. Pure Gaussian with adaptive thresholding
    src = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow("image",image)
    gray = src
    window_name = 'Sobel Demo - Simple Edge Detector'
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # grad_x = cv2.Scharr(gray, ddepth, 0, 1)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray, ddepth, 0, 1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # cv2.imshow(window_name, grad)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # destroys the window showing image

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    proc = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    # using the opening function
    proc = cv2.erode(proc, kernel)
    proc = cv2.dilate(proc, kernel)
    # cv2.imshow("edges", proc)
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image

    # Problems right now:
    # 2. Images are not detected well enough for our use case => Need to be more sensitive
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, threshold1=threshhold_low, threshold2=threshhold_high)

    # cv2.imshow("edges", edges)
    cv2.imwrite("sobel.png", grad)
    cv2.imwrite("blur.png", proc)
    cv2.imwrite("canny.png", edges)
    # preprocessing the image as Laplacian is highly susceptible to noise
    lapl = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    lapl = cv2.Laplacian(lapl, ddepth=cv2.CV_8UC1, ksize=3)
    lapl = lapl * 10
    cv2.imwrite("laplace.png", lapl)

    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image

    contours, hierarchy = cv2.findContours(grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Total Number of Contours found using sobel =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # print(largest_contour)
    contours, hierarchy = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("Total Number of Contours found using blur =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # print(largest_contour)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Total Number of Contours found using canny =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # print(largest_contour)
    contours, hierarchy = cv2.findContours(lapl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Total Number of Contours found using Laplacian =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # print(largest_contour)
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    # Select largest polygon as our square
    points_contour = [largest_contour[top_left][0], largest_contour[top_right][0], largest_contour[bottom_right][0],
                      largest_contour[bottom_left][0]]
    # Warp image to be full screen, to make grid detection straightforward
    top_left, top_right, bottom_right, bottom_left = points_contour[0], points_contour[1], points_contour[2], \
                                                     points_contour[3]
    src2 = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    # searching for the biggest side of the rectangle
    side = max([size_of_side(bottom_right, top_right), size_of_side(top_left, bottom_left),
                size_of_side(bottom_right, bottom_left), size_of_side(top_left, top_right)])
    d = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src2, d)
    image2 = cv2.warpPerspective(image, m, (int(side), int(side)))
    # cv2.imshow("warped_image", image2)
    return image2


def load_digit_classifier():
    print("Loading pretrained model from disk")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded saved model from disk.")

    return loaded_model


def rec_numbers_alt(image, model):
    image = pre_process_image(image.copy(), skip_dilate=True)
    height_of_image, width_of_image = image.shape
    cell_width = int(width_of_image / 9)
    cell_height = int(height_of_image / 9)
    curr_row_pixel = 0
    full_board = []
    for row in range(9):
        curr_col_pixel = 0
        full_row = []
        for col in range(9):
            cell = image[curr_row_pixel:curr_row_pixel + cell_height, curr_col_pixel:curr_col_pixel + cell_width]
            # print(cell.sum())
            if cell.sum() < 150000:
                # print("Unnumbered Cell")
                full_row.append(0)
                # Problem: very, very blurry
                curr_col_pixel += cell_width
                continue
            # src = cv2.GaussianBlur(cell, (3, 3), 0)
            cell = cell[int(0.1 * cell_width):int(cell_width - 0.1 * cell_width),
                   int(0.1 * cell_height):int(cell_height - 0.1 * cell_height)]
            h, w = cell.shape[:2]
            margin = int(np.mean([h, w]) / 2.5)
            # _, bbox, seed = find_largest_feature(cell, [margin, margin], [w - margin, h - margin])
            # cell = cut_from_rect(cell, bbox)

            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            src = cv2.filter2D(cell, -1, sharpen_kernel)
            src = cv2.GaussianBlur(src, (3, 3), 0)
            eros_kernel = np.ones((2, 2), np.uint8)
            src = cv2.dilate(src, eros_kernel, iterations=1)
            _, cell = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
            cell = scale_and_centre(cell, 28, 4)

            # w = bbox[1][0] - bbox[0][0]
            # h = bbox[1][1] - bbox[0][1]
            # if w > 0 and h > 0 and (w * h) > 100 and len(cell) > 0:
            # pass
            #    cell = scale_and_centre(cell, 28, 4)
            # else:
            #    cell = np.zeros((28, 28), np.uint8)
            src = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            prediction = model.predict_classes(src.reshape(1, 1, 28, 28), verbose=0)
            full_row.append(prediction[0])
            cv2.imwrite("CurrentCellTiny.png", src)
            # Problem: very, very blurry
            curr_col_pixel += cell_width
        curr_row_pixel += cell_height
        full_board.append(full_row)
    return image, full_board


def pre_process_image(img, skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    # show_image(inp_img)
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    # show_image(img)
    return img, np.array(bbox, dtype='float32'), seed_point


def show_image(img):
    """Shows an image until any key is pressed"""
    #    print(type(img))
    #    print(img.shape)
    cv2.imshow('image', img)  # Display the image
    #    cv2.imwrite('images/gau_sudoku3.jpg', img)
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows
    return img


def size_of_side(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return np.sqrt((x ** 2) + (y ** 2))


def evaluate_scanned(sudoku, data_file):
    correct_counter = 0
    row = 0
    for rec_row, act_row in zip(sudoku, data_file):
        col = 0
        for rec_num, act_num in zip(rec_row, act_row):
            if rec_num == int(act_num):
                correct_counter += 1
            else:
                sudoku[row][col] = act_num
            col += 1
        row += 1

    corr_pct = round(100 * correct_counter / 81, 2)
    err_pct = 100 - corr_pct
    print("We got {} cells correct, which translates to {} percent correct and {} percent false.".format(correct_counter, corr_pct, err_pct))
    print("To ensure a working solver we changed all the wrong recognitions to their actual number.")
    return


def main():
    images, data_files = load_data_from_folder()
    model = load_digit_classifier()
    for image, data_file in zip(images, data_files):
        warped = recognize_board(image)
        sudoku, sudoku_grid = rec_numbers_alt(warped, model)
        evaluate_scanned(sudoku_grid, data_file)
        # solved_sudoku = solve_sudoku(sudoku)
        # evaluate_result()
    # total_evaluation()


if __name__ == "__main__":
    main()
