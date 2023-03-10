from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import json
import os 
from PIL import Image

from utils import give_labels, preprocess_image_labels, random_name, yolo_points_to_cv2_points, save_image
from augmentation import *



class Worker(QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):

        with open("extras/parameters.json", "r") as f:
            params = json.load(f)

        brightness_interval = params["brightness_interval"]
        contrast_interval = params["contrast_interval"]
        rotate_interval = params["rotate_interval"]
        shearX_interval = params["shearX_interval"]
        shearY_interval = params["shearY_interval"]
        shiftX_interval = params["shiftX_interval"]
        shiftY_interval = params["shiftY_interval"]
        flip = params["flip"]
        brightness_possibility = params["brightness_possibility"]
        contrast_possibility = params["contrast_possibility"]
        rotate_possibility = params["rotate_possibility"]
        shearX_possibility = params["shearX_possibility"]
        shearY_possibility = params["shearY_possibility"]
        shiftX_possibility = params["shiftX_possibility"]
        shiftY_possibility = params["shiftY_possibility"]
        flip_possibility = params["flip_possibility"]
        image_saving_folder = params["image_saving_folder"]
        label_saving_folder = params["label_saving_folder"]
        labels_merged_or_seperated = params["labels_merged_or_seperated"]
        naming_method = params["naming_method"]
        optional_part = params["optional_part"]
        starting_number = params["starting_number"]
        saving_format = params["saving_format"]
        image_paths = params["image_paths"]
        control = params["control"]
        label_control = params["label_control"]
        change_output_image_size = params["change_output_image_size"]

        if control and label_control:
            csv_file = params["csv_file"]
            DataFrame = preprocess_image_labels(csv_file)

        if change_output_image_size:
            output_width = params["output_width"]
            output_height = params["output_height"]

        index = 0
        filenames = []
        region_attributes = []
        bounding_box = []
        k = 100 / len(image_paths)
        
        for image_path in image_paths:
            image = Image.open(image_path)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if change_output_image_size:
                size = (output_width, output_height)
            else:
                size = (image.shape[1], image.shape[0])

            if naming_method == "by indexing":
                save_name = optional_part + str(starting_number) + saving_format
            else:
                save_name = random_name() + saving_format

            if np.random.rand() < brightness_possibility:
                image, _ = img_random_brightness(image, brightness_interval)
            if np.random.rand() < contrast_possibility:
                image, _ = img_random_contrasts(image, contrast_interval)
            if control and label_control:
                try:
                    coor = yolo_points_to_cv2_points(image_path, DataFrame)
                except:
                    print("no label for ", image_path)
                    starting_number += 1
                    index += 1
                    self.progress.emit(int(index*k))
                    continue
                new_coor = coor
                if np.random.rand() < flip_possibility and flip:
                    image, new_coor = img_random_flip(image, new_coor, True)
                if np.random.rand() < rotate_possibility:
                    image, new_coor, _ = img_random_rotate(image, rotate_interval, new_coor, True)
                if np.random.rand() < shearX_possibility:
                    image, new_coor, _ = shear(image, shearX_interval, "X", new_coor, True)
                if np.random.rand() < shearY_possibility:
                    image, new_coor, _ = shear(image, shearY_interval, "Y", new_coor, True)
                if np.random.rand() < shiftX_possibility:
                    image, new_coor, _ = translation(image, shiftX_interval, "X", new_coor, True)
                if np.random.rand() < shiftY_possibility:
                    image, new_coor, _ = translation(image, shiftY_interval, "Y", new_coor, True)
                    
                labels = give_labels(os.path.basename(image_path), DataFrame)
                labels, new_coor = adjust(labels, new_coor)
                for l, new_bb in list(zip(labels, new_coor)):
                    filenames.append(save_name)
                    region_attributes.append(int(l))
                    center_x = new_bb[0][0] * 0.5 + new_bb[1][0] * 0.5
                    center_y = new_bb[0][1] * 0.5 + new_bb[2][1] * 0.5
                    width = np.abs(new_bb[0][0] - new_bb[1][0])
                    height = np.abs(new_bb[0][1] - new_bb[2][1])
                    bounding_box.append((np.round(center_x, 7), np.round(center_y, 7), np.round(width, 7), np.round(height,7)))
                if index % 100 == 99:
                    aug_df = pd.DataFrame({
                        "#filename" : filenames,
                        "region_attributes" : region_attributes,
                        "bounding_box" : bounding_box
                    })
                    try:
                        if labels_merged_or_seperated == "merge labels with original labels":
                            all_df = pd.concat([DataFrame, aug_df])
                            all_df.to_csv(os.path.join(label_saving_folder, "all_labels.csv"), index=False)
                        if labels_merged_or_seperated == "save new labels only":
                            aug_df.to_csv(os.path.join(label_saving_folder, "augmented_image_labels.csv"), index=False)
                    except:
                        if labels_merged_or_seperated == "merge labels with original labels":
                            all_df = pd.concat([DataFrame, aug_df])
                            all_df.to_csv(os.path.join(label_saving_folder, "all_labels_(wait_to_end).csv"), index=False)
                        if labels_merged_or_seperated == "save new labels only":
                            aug_df.to_csv(os.path.join(label_saving_folder, "augmented_image_labels_(wait_to_end).csv"), index=False)
                save_image(image, os.path.join(image_saving_folder, save_name), size)

            else:
                if np.random.rand() < flip_possibility and flip:
                    image = img_random_flip(image)
                if np.random.rand() < rotate_possibility:
                    image, _ = img_random_rotate(image, rotate_interval)
                if np.random.rand() < shearX_possibility:
                    image, _ = shear(image, shearX_interval, "X")
                if np.random.rand() < shearY_possibility:
                    image, _ = shear(image, shearY_interval, "Y")
                if np.random.rand() < shiftX_possibility:
                    image, _ = translation(image, shiftX_interval, "X")
                if np.random.rand() < shiftY_possibility:
                    image, _ = translation(image, shiftY_interval, "Y")
                    
                save_image(image, os.path.join(image_saving_folder, save_name), size)
            
            starting_number += 1
            index += 1
            self.progress.emit(int(index*k))

        if control and label_control:
            aug_df = pd.DataFrame({
                        "#filename" : filenames,
                        "region_attributes" : region_attributes,
                        "bounding_box" : bounding_box
            })
            try:
                if labels_merged_or_seperated == "merge labels with original labels":
                    all_df = pd.concat([DataFrame, aug_df])
                    all_df.to_csv(os.path.join(label_saving_folder, "all_labels.csv"), index=False)
                if labels_merged_or_seperated == "save new labels only":
                    aug_df.to_csv(os.path.join(label_saving_folder, "augmented_image_labels.csv"), index=False)
            except:
                if labels_merged_or_seperated == "merge labels with original labels":
                    all_df = pd.concat([DataFrame, aug_df])
                    all_df.to_csv(os.path.join(label_saving_folder, "all_labels_(wait_to_end).csv"), index=False)
                if labels_merged_or_seperated == "save new labels only":
                    aug_df.to_csv(os.path.join(label_saving_folder, "augmented_image_labels_(wait_to_end).csv"), index=False)

        self.finished.emit()
        
        