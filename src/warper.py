import sys
import cv2 as cv
import json

import processor


# Read config parameters
with open("config.json") as config_file:
    config_vars = json.load(config_file)
    MAX_STRENGTH = config_vars["max_strength"]
    MAX_RANDOM_MOV = config_vars["max_random_mov"]
    CURVED_NORMALIZER = config_vars["curved_normalizer"]
    RESIZE_FACTOR = config_vars["resize_factor"]
    

if __name__ == "__main__":
    # Read image from command line path
    path_read = sys.argv[1]
    img = cv.imread(path_read)

    # Resize image
    img_resize = processor.enlarge_image(img, RESIZE_FACTOR)
    
    # Construct mesh
    mesh = processor.construct_random_mesh(img_resize.shape[0], 
                                           img_resize.shape[1], MAX_RANDOM_MOV, 
                                           MAX_STRENGTH, CURVED_NORMALIZER)
    
    # Transform image
    perturbed_image = processor.apply_warping(img_resize, mesh)
    
    # Save image
    filename_write = sys.argv[2]
    cv.imwrite("assets/" + filename_write, perturbed_image)    