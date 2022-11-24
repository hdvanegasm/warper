import cv2 as cv
import numpy as np


def enlarge_image(img, resize_factor):
    # Enlarge the image
    excess_height = img.shape[1] // resize_factor
    excess_width = img.shape[0] // resize_factor
    img_resize = cv.copyMakeBorder(img, excess_height, excess_height, 
                                   excess_width, excess_width, 
                                   borderType=cv.BORDER_CONSTANT, 
                                   value=(0, 0, 0))
    return img_resize


def construct_random_mesh(width, height, max_rand_mov, max_str, curved_norm):
    """Source code taken and adapted from 
    https://github.com/mhashas/Document-Image-Unwarping-pytorch/issues/7"""
    
    x_axis = np.arange(0, width, 1)
    y_axis = np.arange(0, height, 1)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    # Generates the mesh without modifications, only the points
    perturbed_mesh = np.transpose(np.asarray([x_mesh.flatten("F"), 
                                           y_mesh.flatten("F")]))
    
    n_variations = np.random.randint(max_rand_mov)
    for _ in range(n_variations):
        # Choose a vertex at random
        rnd_point_idx = np.random.randint(perturbed_mesh.shape[0])
        rnd_point = perturbed_mesh[rnd_point_idx, :]
        
        # This vector gives the direction and strength of the deformation
        v = (np.random.rand(1, 2) - 0.5) * max_str
        
        # Compute the distance from every point to the line rnd_point + t * v
        difference = rnd_point - perturbed_mesh
        cross_v_difference = np.cross(v, difference)
        d = np.abs(cross_v_difference) / np.linalg.norm(v, ord=2)
        
        # Choice at random if perform curved or linear deformation
        is_linear = np.random.rand()
        if is_linear < 0.7:
            alpha = np.random.uniform(100, 150)
            w = alpha / (d + alpha)
        else:
            alpha = np.random.uniform(low=1, high=2)
            w = 1 - np.power(d / curved_norm, alpha)
            
        # Resize w to match element wise multiplication
        reshaped_w = w.reshape((w.shape[0], 1))
        duplicated_w = np.append(reshaped_w, reshaped_w, axis=1)
        
        # Make modification to the mesh
        perturbed_mesh = perturbed_mesh + duplicated_w * v
        
    return perturbed_mesh


def apply_warping(img, mesh):
    x_mesh_pert = mesh[:, 1].reshape(img.shape[0], 
                                     img.shape[1]).astype(np.float32)
    y_mesh_pert = mesh[:, 0].reshape(img.shape[0], 
                                    img .shape[1]).astype(np.float32)
    perturbed_image = cv.remap(img, x_mesh_pert, y_mesh_pert,
                               cv.INTER_LINEAR)
    return perturbed_image

    