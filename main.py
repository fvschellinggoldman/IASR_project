# This is the main runner function from which the code is started
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import operator
import sys
from PIL import Image
import imutils

def load_data_from_folder(folder_location="."):
    #indicates that directory path is being given
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


def preprocess_data(cell_content):
    #function is used to preprocess each cell segment before extracting it's feature
    img = cell_content
    sizes= np.shape(img)
    img = imutils.resize(img,height=600)
    proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(proc, (39,39), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 2.5)
    kernel = np.array([[0.,0.,0., 1.,0., 0.,0.],[0.,0.,1., 1., 1.,0.,0.],[0.,1.,1., 1., 1.,1.,0.], [1., 1., 1.,1.,1.,1.,1.],[0., 1., 1.,1.,1.,1.,0.], [0.,0.,1., 1., 1.,0.,0.],[0.,0.,0., 1., 0.,0.,0.]], np.uint8)
    edges = cv2.Canny(proc, threshold1=30, threshold2=60)
    #edges = cv2.Laplacian(proc, ddepth=cv2.CV_8UC1, ksize=3)
    proc = cv2.bitwise_not(proc)
    proc = cv2.dilate(proc, kernel,iterations=1)
    for i in range(10):
        proc = cv2.dilate(proc, kernel)
        proc = cv2.erode(proc, kernel)
    proc = cv2.erode(proc, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print("Total Number of Contours found using Canny =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    sizes2=np.shape(proc)
    im_out = np.ones(sizes2)
    im_out=cv2.fillConvexPoly(im_out, largest_contour, 0)
    proc = cv2.bitwise_not(proc)
    for i in range(sizes2[0]):
       for j in range(sizes2[1]):
           if (im_out[i][j]==proc[i][j]):
                  im_out[i][j]=1
           else:
                  im_out[i][j]=0

   # return imutils.resize(im_out,height=sizes[0])
    return im_out



def recognize_board(image, threshhold_low=30, threshhold_high=60):
    #Other edge detection 1. Gauss + Sobel/Scharr 2. Pure Gaussian with adaptive thresholding
    src = cv2.GaussianBlur(image, (3, 3), 0)
    #cv2.imshow("image",image)
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
    #using the opening function
    proc = cv2.erode(proc, kernel)
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
    #preprocessing the image as Laplacian is highly susceptible to noise
    lapl = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    lapl = cv2.Laplacian(lapl, ddepth=cv2.CV_8UC1, ksize=3)
    lapl = lapl * 10
    cv2.imwrite("laplace.png", lapl)


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
    contours, hierarchy = cv2.findContours(lapl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Total Number of Contours found using Laplacian =", len(contours))
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # print(largest_contour)
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in largest_contour]), key=operator.itemgetter(1))
    # Select largest polygon as our square
    points_contour=[largest_contour[top_left][0], largest_contour[top_right][0], largest_contour[bottom_right][0], largest_contour[bottom_left][0]]
    # Warp image to be full screen, to make grid detection straightforward
    top_left, top_right, bottom_right, bottom_left = points_contour[0], points_contour[1], points_contour[2], points_contour[3]
    src2 = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    #searching for the biggest side of the rectangle
    side = max([size_of_side(bottom_right, top_right),size_of_side(top_left, bottom_left),size_of_side(bottom_right, bottom_left),size_of_side(top_left, top_right)])
    d = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src2, d)
    image2=cv2.warpPerspective(image, m, (int(side), int(side)))
    cv2.imshow("warped_image",image2)
    return image2

def recognize_cells(image):
    sudoku = image
    # we have to infer 81 cell grids from a sudoku image
    cells = []
    # .shape gives us a number of pixels in a row
    length = sudoku.shape[:1]
    # we get the size of each cell
    length = length[0] / 9
    for j in range(9):
        for i in range(9):
            top_left = (i * length, j * length)
            bottom_right = ((i + 1) * length, (j + 1) * length)
            cells.append((top_left, bottom_right))
    return cells

def recognize_numbers(image):
    sudoku = image
    cells=recognize_cells(sudoku)
    digits = []
    #img = pre_process_image(img.copy(), skip_dilate=True)
    for cell in cells:
        digits.append(extract_digit(cell,sudoku))
        #wait = input("Press Enter to continue.")#
    cv2.imshow("",digits[2])
    return digits

def extract_digit(cell,image):
    bounds = cell
    img = image
    #cut_image gets each individual cell image
    digit = cut_image(img, bounds)
    #digit = Image.fromarray(digit, 'RGB')
    digit = preprocess_data(digit)
    digit =scale_and_center(digit)
    return digit
def scale_and_center(digit):
    #center of the digit
    d1=int(np.shape(digit)[0]/2)
    d2=int(np.shape(digit)[1]/2)
    x_sum=0
    y_sum=0
    x_ctr=0
    y_ctr=0
    digit2 = np.zeros(np.shape(digit))
    for i in range(np.shape(digit)[0]):
        for j in range(np.shape(digit)[1]):
            if (digit[i][j]==1):
                x_sum=x_sum+i
                y_sum=y_sum+j
                x_ctr=x_ctr+1
                y_ctr=y_ctr+1
    if (x_ctr != 0 and y_ctr != 0):
        x_centroid = int(x_sum/x_ctr)
        y_centroid = int(y_sum/y_ctr)
        shift_x = int(x_centroid-d1)
        shift_y = int(y_centroid-d2)
        for i in range(2*d1):
            for j in range(2*d2):
                if (i+shift_x < np.shape(digit)[0] and j+shift_y < np.shape(digit)[1]):
                    digit2[i+shift_x][j+shift_y]=digit[i][j]
    #scaling still has to be done
    return digit2

def cut_image(image,bounds):
    img = image
    width = int(bounds[1][0] - bounds[0][0])
    height = int(bounds[1][1] - bounds[0][1])
    img2 = np.ones((height-10, width-10, 3), dtype=np.uint8)
    pi = int(bounds[0][0])
    pj = int(bounds[0][1])
    for i in range(5,width-5):
        for j in range(5,height-5):
            img2[i-5,j-5]=img[i+pi][j+pj]
    #cv2.imshow("cell",img2)
    return img2

def size_of_side(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return np.sqrt((x ** 2) + (y ** 2))

def main():
    images = load_data_from_folder()
    #images = preprocess_data(images)
    for image in images:
        warped=recognize_board(image)
        sudoku = recognize_numbers(warped)
        #solved_sudoku = solve_sudoku(sudoku)
        #evaluate_result()
    #total_evaluation()


if __name__ == "__main__":
    main()