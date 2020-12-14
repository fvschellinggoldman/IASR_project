# This is the main runner function from which the code is started
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def load_data_from_folder(folder_location="."):
    # This for now will be a prototype function to load a single image
    images_data = []
    for filename in os.listdir(folder_location):
        if filename.endswith(".jpg"):
            print(filename)
            images_data.append(cv2.imread(os.path.join(folder_location, filename), cv2.IMREAD_GRAYSCALE))
            #cv2.imshow("t", images_data[0])
        else:
            continue
    return images_data


def preprocess_data(original_images):
    # For now this function does nothing, later on it will maybe rotate etc.
    return original_images


def recognize_board(image, threshhold_low=30, threshhold_high=60):
    #Other edge detection 1. Gauss + Sobel/Scharr 2. Pure Gaussian with adaptive thresholding
    src = cv2.GaussianBlur(image, (3, 3), 0)

    gray = src
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    #grad_x = cv2.Scharr(gray, ddepth, 0, 1)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    #grad_y = cv2.Scharr(gray, ddepth, 0, 1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    #cv2.imshow(window_name, grad)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  # destroys the window showing image

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    proc = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)
    #cv2.imshow("edges", proc)
    #cv2.waitKey(0)  # waits until a key is pressed
    #cv2.destroyAllWindows()  # destroys the window showing image

    # Problems right now:
    # 2. Images are not detected well enough for our use case => Need to be more sensitive
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, threshold1=threshhold_low, threshold2=threshhold_high)

    #cv2.imshow("edges", edges)
    cv2.imwrite("sobel.png", grad)
    cv2.imwrite("blur.png", proc)
    cv2.imwrite("canny.png", edges)

    #cv2.waitKey(0)  # waits until a key is pressed
    #cv2.destroyAllWindows()  # destroys the window showing image

    contours, hierarchy = cv2.findContours(grad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Total Number of Contours found using sobel =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    #print(largest_contour)
    contours, hierarchy = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Total Number of Contours found using blur =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    #print(largest_contour)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Total Number of Contours found using canny =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    #print(largest_contour)

    # Select largest polygon as our square
    # Warp image to be full screen, to make grid detection straightforward
    return image


def recognize_numbers(image):
    sudoku = image
    return sudoku


def main():
    images = load_data_from_folder()
    images = preprocess_data(images)
    for image in images:
        recognize_board(image)
        #sudoku = recognize_numbers()
        #solved_sudoku = solve_sudoku(sudoku)
        #evaluate_result()
    #total_evaluation()


if __name__ == "__main__":
    main()