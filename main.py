# This is the main runner function from which the code is started
import cv2
import numpy as np
import os
import operator
from keras.models import model_from_json

import Solver


def load_data_from_folder(folder_location="."):
    data_dict = {}
    image_dict = {}
    file_ids = []
    for filename in os.listdir(folder_location):
        if filename.endswith(".jpg"):
            file_id = filename[:-3]
            file_ids.append(file_id)
            print("Working with {}.".format(filename))
            image_dict[file_id] = cv2.imread(os.path.join(folder_location, filename), cv2.IMREAD_GRAYSCALE)
            # cv2.imshow("t", images_data[0])
        elif filename.endswith(".dat"):
            file_id = filename[:-3]
            print("Working with {}.".format(filename))
            with open(os.path.join(folder_location, filename)) as f:
                cleaned_data = [i.strip().split(" ") for i in f.readlines()[2:]]
                data_dict[file_id] = cleaned_data
        else:
            continue
    return file_ids, data_dict, image_dict


def recognize_board(image, threshhold_low=30, threshhold_high=60):
    # Other edge detection 1. Gauss + Sobel/Scharr 2. Pure Gaussian with adaptive thresholding
    src = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow("image",image)
    gray = src
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

    proc = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    # using the opening function
    proc = cv2.erode(proc, kernel)
    proc = cv2.dilate(proc, kernel)

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
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    contours, hierarchy = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    contours, hierarchy = cv2.findContours(lapl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
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
    cv2.imwrite("warped_image.png", image2)
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
    image = pre_process_image(image.copy())
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

            cell = cell[int(0.1 * cell_width):int(cell_width - 0.1 * cell_width),
                   int(0.1 * cell_height):int(cell_height - 0.1 * cell_height)]
            if cell[20:28, 20:28].sum() < 0.3 * (255 * 16):  # more than 30 percent of the center is filled
                # print("Unnumbered Cell")
                full_row.append(0)
                curr_col_pixel += cell_width
                continue

            src = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            #show_image(src)
            prediction = model.predict_classes(src.reshape(1, 1, 28, 28), verbose=0)
            full_row.append(prediction[0])
            cv2.imwrite("CurrentCellTiny.png", src)
            curr_col_pixel += cell_width
        curr_row_pixel += cell_height
        full_board.append(full_row)
    return image, full_board


def pre_process_image(img):
    img_out = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_out = cv2.bitwise_not(img_out, img_out)

    return img_out


def show_image(img):
    cv2.imshow('image', img)  # Display the image
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
                sudoku[row][col] = int(act_num)
            col += 1
        row += 1

    corr_pct = round(100 * correct_counter / 81, 2)
    err_pct = round(100 - corr_pct, 2)
    print("We got {} cells correct, which translates to {} percent correct and {} percent false.".format(correct_counter, corr_pct, err_pct))
    print("To ensure a working solver we changed all the wrong recognitions to their actual number.")
    return corr_pct


def main():
    file_ids, data_dict, image_dict = load_data_from_folder()
    model = load_digit_classifier()
    correct_data = []
    for file_id in file_ids:
        image = image_dict[file_id]
        data = data_dict[file_id]
        warped = recognize_board(image)
        sudoku, sudoku_grid = rec_numbers_alt(warped, model)
        correct_data.append(evaluate_scanned(sudoku_grid, data))
        if Solver.sudoku_solve(sudoku_grid):
            Solver.Sudoku_grid(sudoku_grid)
        else:
            print("No solution exists")
    # total_evaluation() --> performed manually with output from below
    print(correct_data)
    print(file_ids)


if __name__ == "__main__":
    main()
