from PIL import Image
import cv2
import os

import pandas as pd
import numpy as np


def convert_data(x):
    a = x.split(")")[0].split("(")[1].split(",")
    try:
        return (int(a[0].strip()), int(a[1].strip()), int(a[2].strip()), int(a[3].strip()))
    except:
        return (float(a[0].strip()), float(a[1].strip()), float(a[2].strip()), float(a[3].strip()))


def give_labels(image, df):
    try:
        labels = [ int(i) for i in df[df["#filename"] == os.path.basename(image)]["region_attributes"] ]
    except:
        labels = [ i for i in df[df["#filename"] == os.path.basename(image)]["region_attributes"] ]
    return labels


def preprocess_image_labels(label_file):

    df = pd.read_csv(label_file)
    df["bounding_box"] = df["bounding_box"].apply(convert_data)
    return df


def yolo_points_to_cv2_points(image, df):
    
    image = os.path.basename(image)
    bb = np.array([ list(i) for i in df[df["#filename"] == image]["bounding_box"] ])

    left_up = np.array([bb[:,0] - bb[:,2] / 2, bb[:,1] - bb[:,3] / 2]).T
    right_down = np.array([bb[:,0] + bb[:,2] / 2, bb[:,1] + bb[:,3] / 2]).T
    left_down = np.array([bb[:,0] - bb[:,2] / 2, bb[:,1] + bb[:,3] / 2]).T
    right_up = np.array([bb[:,0] + bb[:,2] / 2, bb[:,1] - bb[:,3] / 2]).T

    coor = np.hstack([left_up, right_up, right_down, left_down]).reshape(bb.shape[0], 4, 2)
    return coor


def Give_Pixmap(filename, final_width, final_height, bounding_boxes=None, labels=None, on_aug_img=False):

    if on_aug_img:
        image = Image.open("extras/augmented_image.jpg")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    else:
        image = Image.open(filename)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("extras/original_image.jpg", image)
    
    imagename = "extras/original_image.jpg"
    img = Image.open(imagename)
    info_dict = {
        "Image Size": img.size,
        "Image Height": img.height,
        "Image Width": img.width,
        "Image Format": img.format,
        "Image Mode": img.mode,
        "Image is Animated": getattr(img, "is_animated", False),
        "Frames in Image": getattr(img, "n_frames", 1),
    }
    info = ""
    for label,value in info_dict.items():
        info = info + f"{label:25}: {value}\n"


    height, width = image.shape[:2]
    if height > width:
        ratio = final_height / height
        image_height = final_height
        image_width = int(width * ratio)
    else:
        ratio = final_width / width
        image_width = final_width
        image_height = int(height * ratio)
    
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_LINEAR)

    try:
        if on_aug_img:
            coor = bounding_boxes
        else:
            coor = yolo_points_to_cv2_points(filename, bounding_boxes)
            labels = give_labels(filename, bounding_boxes)

        color = (0, 0, 255)

        thickness = 1

        start_points = (coor[:,0,:] * np.array([image.shape[1], image.shape[0]])).astype("int")
        end_points = (coor[:,2,:] * np.array([image.shape[1], image.shape[0]])).astype("int")
        for i in range(coor.shape[0]):
            image = cv2.rectangle(image, tuple(start_points[i]), tuple(end_points[i]), color, thickness)
            image = cv2.putText(image, str(labels[i]), (start_points[i][0], start_points[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        if on_aug_img:
            cv2.imwrite("extras/resized_augmented_image.jpg", image)
            return True
        else:
            cv2.imwrite("extras/resized_original_image.jpg", image)
            return True, info

    except:
        if on_aug_img:
            cv2.imwrite("extras/resized_augmented_image.jpg", image)
            return True
        else:
            cv2.imwrite("extras/resized_original_image.jpg", image)
            return True, info

    
def save_image(image, save_path, size):
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(image)
    image.save(save_path)


def random_name():
    chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","r","s","t","u","v","x","y","z"]
    numbers = ["0","1","2","3","4","5","6","7","8","9"]
    chars.extend(numbers)
    name = ""
    for _ in range(8):
        name += np.random.choice(chars)
    name += "-"
    for _ in range(4):
        name += np.random.choice(chars)
    name += "-"
    for _ in range(4):
        name += np.random.choice(chars)
    name += "-"
    for _ in range(12):
        name += np.random.choice(chars)
    return name

    