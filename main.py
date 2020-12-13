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
            images_data.append(cv2.imread(os.path.join(folder_location, filename)))
        else:
            continue
    return images_data


def preprocess_data(original_images):
    # For now this function does nothing, later on it will maybe rotate etc.
    return original_images


def recognize_board(image):
    # Problems right now:
    # 1. It only sometimes detects edges??
    # 2. Images are not detected well enough for our use case => Need to be more sensitive
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, threshold1=30, threshold2=100)

    cv2.imshow("edges", edges)

    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image
    # Select largest polygon as our square
    # Warp image to be full screen
    return image


def main():
    images = load_data_from_folder()
    images = preprocess_data(images)
    for image in images:
        recognize_board(image)
        #recognize_numbers()
        #solve_sudoku()
        #evaluate_result()
    #total_evaluation()


if __name__ == "__main__":
    main()