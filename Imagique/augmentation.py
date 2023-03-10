import numpy as np
import cv2



def img_random_brightness(image, range):
    min, max = range
    random_brightness = int(np.random.uniform(min, max))
    image = cv2.convertScaleAbs(image, alpha=1, beta=random_brightness)
    return image, random_brightness


def img_random_contrasts(image, range):
    min, max = range
    random_contrast = np.random.uniform(min, max)
    image = cv2.convertScaleAbs(image, alpha=random_contrast, beta=0)
    return image, random_contrast


def img_random_rotate(image, range, coor=None, control=False):
    min, max = range
    random_degree = -np.random.uniform(min, max)
    height = image.shape[1]
    width = image.shape[0]
    center = (int(height*(1/2)), int(width*(1/2)))
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=random_degree, scale=1-np.abs(random_degree)*0.012)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (height, width))
    if control == True:
        new_coor = np.zeros((1,4,2))
        for bb in coor:
            bb = (bb * np.array([image.shape[1], image.shape[0]])).astype("int")
            ones = np.ones(shape=(len(bb), 1))
            points_ones = np.hstack([bb, ones])
            transformed_points = rotation_matrix.dot(points_ones.T).T
            transformed_points = transformed_points / np.array([image.shape[1], image.shape[0]])
            x_min = transformed_points[0][0] * 0.5 + transformed_points[3][0] * 0.5
            x_max = transformed_points[1][0] * 0.5 + transformed_points[2][0] * 0.5
            y_min = transformed_points[0][1] * 0.5 + transformed_points[1][1] * 0.5
            y_max = transformed_points[2][1] * 0.5 + transformed_points[3][1] * 0.5
            left_up = np.array([x_min, y_min]).T
            right_up = np.array([x_max, y_min]).T
            right_down = np.array([x_max, y_max]).T
            left_down = np.array([x_min, y_max]).T
            new_bb = np.hstack([left_up, right_up, right_down, left_down]).reshape(1,4,2)
            new_coor = np.vstack([new_coor, new_bb])
        new_coor = new_coor[1:,:,:]
        return rotated_image, new_coor, -random_degree
    else:
        return rotated_image, random_degree


def shear(image, range, direction, coor=None, control=False):
    min, max = range
    random_shear = np.random.uniform(min, max)
    height = image.shape[1]
    width = image.shape[0]
    if direction == "X":
        M = np.float32([[1,random_shear,0],
                        [0,1,0],
                        [0,0,1]])
    elif direction == "Y":
        M = np.float32([[1,0,0],
                        [random_shear,1,0],
                        [0,0,1]])
    else:
        print("wrong direction")
        return
    M[0,2] = -M[0,1] * width/2
    M[1,2] = -M[1,0] * height/2
    sheared_img = cv2.warpPerspective(image, M, (height, width))
    if control == True:
        new_coor = np.zeros((1,4,2))
        for bb in coor:
            bb = (bb * np.array([image.shape[1], image.shape[0]])).astype("int")
            ones = np.ones(shape=(len(bb), 1))
            points_ones = np.hstack([bb, ones])
            transformed_points = transformed_points = M.dot(points_ones.T).T
            transformed_points = transformed_points / np.array([image.shape[1], image.shape[0], 1])
            x_min = transformed_points[0][0] * 0.5 + transformed_points[3][0] * 0.5
            x_max = transformed_points[1][0] * 0.5 + transformed_points[2][0] * 0.5
            y_min = transformed_points[0][1] * 0.5 + transformed_points[1][1] * 0.5
            y_max = transformed_points[2][1] * 0.5 + transformed_points[3][1] * 0.5
            left_up = np.array([x_min, y_min]).T
            right_up = np.array([x_max, y_min]).T
            right_down = np.array([x_max, y_max]).T
            left_down = np.array([x_min, y_max]).T
            new_bb = np.hstack([left_up, right_up, right_down, left_down]).reshape(1,4,2)
            new_coor = np.vstack([new_coor, new_bb])
        new_coor = new_coor[1:,:,:]
        return sheared_img, new_coor, random_shear
    else:
        return sheared_img, random_shear


def translation(image, range, direction, coor=None, control=False):
    min, max = range
    random_translation = np.random.uniform(min, max)
    height = image.shape[1]
    width = image.shape[0]
    if direction == "X":
        M = np.float32([[1,0,int(width * random_translation)],
                        [0,1,0],
                        [0,0,1]])
    elif direction == "Y":
        M = np.float32([[1,0,0],
                        [0,1,int(height * random_translation)],
                        [0,0,1]])
    else:
        print("wrong direction")
        return
    translated_image = cv2.warpPerspective(image,M,(height,width))
    if control:
        new_coor = np.zeros((1,4,2))
        for bb in coor:
            bb = (bb * np.array([image.shape[1], image.shape[0]])).astype("int")
            ones = np.ones(shape=(len(bb), 1))
            points_ones = np.hstack([bb, ones])
            transformed_points = transformed_points = M.dot(points_ones.T).T
            transformed_points = transformed_points / np.array([image.shape[1], image.shape[0], 1])
            x_min = transformed_points[0][0] * 0.5 + transformed_points[3][0] * 0.5
            x_max = transformed_points[1][0] * 0.5 + transformed_points[2][0] * 0.5
            y_min = transformed_points[0][1] * 0.5 + transformed_points[1][1] * 0.5
            y_max = transformed_points[2][1] * 0.5 + transformed_points[3][1] * 0.5
            left_up = np.array([x_min, y_min]).T
            right_up = np.array([x_max, y_min]).T
            right_down = np.array([x_max, y_max]).T
            left_down = np.array([x_min, y_max]).T
            new_bb = np.hstack([left_up, right_up, right_down, left_down]).reshape(1,4,2)
            new_coor = np.vstack([new_coor, new_bb])
        new_coor = new_coor[1:,:,:]
        return translated_image, new_coor, random_translation
    else:
        return translated_image, random_translation


def img_random_flip(image, coor=None, control=False):
    image = cv2.flip(image,1)
    if control:
        x = coor[:,:,0]
        y = coor[:,:,1]
        new_x = np.ones(x.shape) - x
        new_coor = np.hstack([new_x.reshape(x.shape[0] * x.shape[1], 1), y.reshape(y.shape[0] * y.shape[1], 1)]).reshape(coor.shape)
        return image, new_coor
    else:
        return image


def adjust(labels ,coors):
    old_coors = coors.copy()
    new_coors = np.zeros((1,4,2))

    for label,bb,bb_old in list(zip(labels, coors, old_coors)):
        for corner in bb:
            if corner[0] >= 1:
                corner[0] = 0.995
            if corner[1] >= 1:
                corner[1] = 0.995
            if corner[0] <= 0:
                corner[0] = 0.005
            if corner[1] <= 0:
                corner[1] = 0.005
        
        old_width = np.abs((bb_old[2][1] - bb_old[1][1]) * 1000) 
        old_height = np.abs((bb_old[1][0] - bb_old[0][0]) * 1000)
        old_area = old_height * old_width
        
        width = np.abs((bb[2][1] - bb[1][1]) * 1000)
        height = np.abs((bb[1][0] - bb[0][0]) * 1000)
        area = height * width

        if area < old_area*0.4:
            labels.remove(label)
        else:
            new_coors = np.vstack([new_coors, bb.reshape(1,4,2)])
            
    new_coors = new_coors[1:,:,:]
    
    return labels, new_coors
