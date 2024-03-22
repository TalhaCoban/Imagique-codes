
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog
from PyQt5.QtWidgets import QLabel, QScrollArea, QSlider, QCheckBox, QComboBox, QGroupBox, QListWidget, QListWidgetItem, QPushButton, QLineEdit, QProgressBar
from qtrangeslider import QRangeSlider

import sys
import os
import numpy as np
import json
from PIL import Image
import time

from utils import Give_Pixmap, give_labels, preprocess_image_labels, yolo_points_to_cv2_points
from augmentation import *
from worker_class import Worker
from open_table_class import OpenCSVApp
from show_table_class import ShowCSVApp



class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1000, 700)

        self.original_image_height = 736
        self.original_image_width = 832
        self.image_name = 0
        self.image_names = 0
        self.label_control = 0

        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu("File")
        self.import_menu = self.menubar.addMenu("import")

        self.open_image_action = self.file_menu.addAction("Open images")
        self.open_image_action.triggered.connect(self.open_image)

        self.import_labels_csv = self.import_menu.addAction("import csv")
        self.import_labels_csv.triggered.connect(self.get_csv_path)

        self.Widgets()
        self.Layouts()
        self.update_interval()


    def Widgets(self):

        # original image region -- left layout
        self.previous_image_button = QPushButton("<")
        self.previous_image_button.setFont(QFont("Times", 15, QFont.Bold))
        self.previous_image_button.setMaximumWidth(42)
        self.previous_image_button.setDisabled(True)
        self.previous_image_button.clicked.connect(self.get_previous_image)
        self.size_button_smaller = QPushButton("-")
        self.size_button_smaller.setFont(QFont("Times", 15, QFont.Bold))
        self.size_button_smaller.setMaximumWidth(42)
        self.size_button_smaller.setDisabled(True)
        self.size_button_smaller.clicked.connect(self.image_get_smaller)
        self.size_button_bigger = QPushButton("+")
        self.size_button_bigger.setFont(QFont("Times", 15, QFont.Bold))
        self.size_button_bigger.setMaximumWidth(42)
        self.size_button_bigger.setDisabled(True)
        self.size_button_bigger.clicked.connect(self.image_get_bigger)
        self.next_image_button = QPushButton(">")
        self.next_image_button.setFont(QFont("Times", 15, QFont.Bold))
        self.next_image_button.setMaximumWidth(42)
        self.next_image_button.setDisabled(True)
        self.next_image_button.clicked.connect(self.get_next_image)

        # /////original image region -- left layout
        # adjustments-- left layout
        ## intervals
        self.interval_brightness_value_label = QLabel("interval of brightness value : ")
        self.interval_contrast_value_label = QLabel("interval of brightness value : ")
        self.interval_rotate_value_label = QLabel("interval of rotation angle : ")
        self.interval_shearX_value_label = QLabel("interval of shearing X value : ")
        self.interval_shearY_value_label = QLabel("interval of Shearing Y value : ")
        self.interval_shiftX_value_label = QLabel("interval of Shifting X value : ")
        self.interval_shiftY_value_label = QLabel("interval of Shifting Y value : ")
        self.flip_label = QLabel("Flip")

        self.interval_brightness_value = QRangeSlider(Qt.Horizontal)
        self.interval_brightness_value.setMinimum(0)
        self.interval_brightness_value.setMaximum(200)
        self.interval_brightness_value.setValue([40,160])
        self.interval_brightness_value.setTickInterval(10)
        self.interval_brightness_value.setTickPosition(QSlider.TicksBelow)
        self.interval_brightness_value.valueChanged.connect(self.update_interval)

        self.interval_contrast_value = QRangeSlider(Qt.Horizontal)
        self.interval_contrast_value.setMinimum(0)
        self.interval_contrast_value.setMaximum(15)
        self.interval_contrast_value.setValue([4,11])
        self.interval_contrast_value.setTickInterval(1)
        self.interval_contrast_value.setTickPosition(QSlider.TicksBelow)
        self.interval_contrast_value.valueChanged.connect(self.update_interval)

        self.interval_rotate_value = QRangeSlider(Qt.Horizontal)
        self.interval_rotate_value.setMinimum(0)
        self.interval_rotate_value.setMaximum(20)
        self.interval_rotate_value.setValue([4,16])
        self.interval_rotate_value.setTickInterval(1)
        self.interval_rotate_value.setTickPosition(QSlider.TicksBelow)
        self.interval_rotate_value.valueChanged.connect(self.update_interval)

        self.interval_shearX_value = QRangeSlider(Qt.Horizontal)
        self.interval_shearX_value.setMinimum(0)
        self.interval_shearX_value.setMaximum(40)
        self.interval_shearX_value.setValue([10,30])
        self.interval_shearX_value.setTickInterval(5)
        self.interval_shearX_value.setTickPosition(QSlider.TicksBelow)
        self.interval_shearX_value.valueChanged.connect(self.update_interval)

        self.interval_shearY_value = QRangeSlider(Qt.Horizontal)
        self.interval_shearY_value.setMinimum(0)
        self.interval_shearY_value.setMaximum(40)
        self.interval_shearY_value.setValue([10,30])
        self.interval_shearY_value.setTickInterval(5)
        self.interval_shearY_value.setTickPosition(QSlider.TicksBelow)
        self.interval_shearY_value.valueChanged.connect(self.update_interval)

        self.interval_shiftX_value = QRangeSlider(Qt.Horizontal)
        self.interval_shiftX_value.setMinimum(0)
        self.interval_shiftX_value.setMaximum(40)
        self.interval_shiftX_value.setValue([10,30])
        self.interval_shiftX_value.setTickInterval(5)
        self.interval_shiftX_value.setTickPosition(QSlider.TicksBelow)
        self.interval_shiftX_value.valueChanged.connect(self.update_interval)

        self.interval_shiftY_value = QRangeSlider(Qt.Horizontal)
        self.interval_shiftY_value.setMinimum(0)
        self.interval_shiftY_value.setMaximum(40)
        self.interval_shiftY_value.setValue([10,30])
        self.interval_shiftY_value.setTickInterval(5)
        self.interval_shiftY_value.setTickPosition(QSlider.TicksBelow)
        self.interval_shiftY_value.valueChanged.connect(self.update_interval)

        self.flip_bool = QSlider(Qt.Horizontal)
        self.flip_bool.setMinimum(0)
        self.flip_bool.setMaximum(1)
        self.flip_bool.setValue(1)
        self.flip_bool.setTickPosition(QSlider.TicksBelow)
        self.flip_bool.valueChanged.connect(self.update_interval)
        ## ///intervals
        ## possibilities
        self.possibility_brightness_value_label = QLabel("possibility of brightness adjustment : ")
        self.possibility_contrast_value_label = QLabel("possibility of brightness adjustment: ")
        self.possibility_rotate_value_label = QLabel("possibility of rotation adjustment : ")
        self.possibility_shearX_value_label = QLabel("possibility of shearing X adjustment : ")
        self.possibility_shearY_value_label = QLabel("possibility of Shearing Y adjustment : ")
        self.possibility_shiftX_value_label = QLabel("possibility of Shifting X adjustment : ")
        self.possibility_shiftY_value_label = QLabel("possibility of Shifting Y adjustment : ")
        self.possibility_flip_label = QLabel("possibility of flipping : ")

        self.possibility_brightness_value = QSlider(Qt.Horizontal)
        self.possibility_brightness_value.setMinimum(0)
        self.possibility_brightness_value.setMaximum(100)
        self.possibility_brightness_value.setValue(50)
        self.possibility_brightness_value.setTickInterval(10)
        self.possibility_brightness_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_brightness_value.valueChanged.connect(self.update_interval)

        self.possibility_contrast_value = QSlider(Qt.Horizontal)
        self.possibility_contrast_value.setMinimum(0)
        self.possibility_contrast_value.setMaximum(100)
        self.possibility_contrast_value.setValue(50)
        self.possibility_contrast_value.setTickInterval(10)
        self.possibility_contrast_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_contrast_value.valueChanged.connect(self.update_interval)

        self.possibility_rotate_value = QSlider(Qt.Horizontal)
        self.possibility_rotate_value.setMinimum(0)
        self.possibility_rotate_value.setMaximum(100)
        self.possibility_rotate_value.setValue(50)
        self.possibility_rotate_value.setTickInterval(10)
        self.possibility_rotate_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_rotate_value.valueChanged.connect(self.update_interval)

        self.possibility_shearX_value = QSlider(Qt.Horizontal)
        self.possibility_shearX_value.setMinimum(0)
        self.possibility_shearX_value.setMaximum(100)
        self.possibility_shearX_value.setValue(50)
        self.possibility_shearX_value.setTickInterval(10)
        self.possibility_shearX_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_shearX_value.valueChanged.connect(self.update_interval)

        self.possibility_shearY_value = QSlider(Qt.Horizontal)
        self.possibility_shearY_value.setMinimum(0)
        self.possibility_shearY_value.setMaximum(100)
        self.possibility_shearY_value.setValue(50)
        self.possibility_shearY_value.setTickInterval(10)
        self.possibility_shearY_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_shearY_value.valueChanged.connect(self.update_interval)

        self.possibility_shiftX_value = QSlider(Qt.Horizontal)
        self.possibility_shiftX_value.setMinimum(0)
        self.possibility_shiftX_value.setMaximum(100)
        self.possibility_shiftX_value.setValue(50)
        self.possibility_shiftX_value.setTickInterval(10)
        self.possibility_shiftX_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_shiftX_value.valueChanged.connect(self.update_interval)

        self.possibility_shiftY_value = QSlider(Qt.Horizontal)
        self.possibility_shiftY_value.setMinimum(0)
        self.possibility_shiftY_value.setMaximum(100)
        self.possibility_shiftY_value.setValue(50)
        self.possibility_shiftY_value.setTickInterval(10)
        self.possibility_shiftY_value.setTickPosition(QSlider.TicksBelow)
        self.possibility_shiftY_value.valueChanged.connect(self.update_interval)

        self.possibility_flip = QSlider(Qt.Horizontal)
        self.possibility_flip.setMinimum(0)
        self.possibility_flip.setMaximum(100)
        self.possibility_flip.setValue(50)
        self.possibility_flip.setTickInterval(10)
        self.possibility_flip.setTickPosition(QSlider.TicksBelow)
        self.possibility_flip.valueChanged.connect(self.update_interval)
        ## ///possibilities
        # other widgets - bottom left region
        self.bounding_boxes_checkbox = QCheckBox("Apply augmentation to bounding boxes")
        self.bounding_boxes_checkbox.setChecked(False)
        self.augmentation_selected_image_label = QLabel("Apply Augmentation to Selected image")
        self.augmentation_selected_image_button = QPushButton("Apply")
        self.augmentation_selected_image_button.setDisabled(True)
        self.augmentation_selected_image_button.clicked.connect(self.apply_augmentation_to_selected_image)
        self.show_table_button = QPushButton("Show Table")
        self.show_table_button.setDisabled(True)
        self.show_table_button.clicked.connect(self.show_table_function)

        ## augmenteatin to all images
        self.save_image_folder = QLineEdit()
        self.save_image_folder.setPlaceholderText("save output images to ...")
        self.save_image_folder.setMinimumWidth(300)

        self.change_output_size_checkbox = QCheckBox()
        self.change_output_size_checkbox.clicked.connect(self.change_output_size_checkbox_function)
        self.change_output_size_checkbox.setChecked(False)

        self.output_height = QLineEdit()
        self.output_height.setPlaceholderText("output height")
        self.output_height.setMinimumHeight(26)
        self.output_height.setMinimumWidth(115)
        self.output_height.setMaximumWidth(115)
        self.output_height.setDisabled(True)

        self.output_width = QLineEdit()
        self.output_width.setPlaceholderText("output width")
        self.output_width.setMinimumHeight(26)
        self.output_width.setMinimumWidth(115)
        self.output_width.setMaximumWidth(115)
        self.output_width.setDisabled(True)

        self.save_image_folder_button = QPushButton("Search")
        self.save_image_folder_button.clicked.connect(self.select_image_saving_folder)

        self.save_df_folder = QLineEdit()
        self.save_df_folder.setPlaceholderText("save output labels to ...")
        self.save_df_folder.setMinimumWidth(300)

        self.save_df_folder_button = QPushButton("Search")
        self.save_df_folder_button.clicked.connect(self.select_df_saving_folder)

        self.labels_merged_or_seperated = QComboBox()
        self.labels_merged_or_seperated.addItems(["merge labels with original labels", "save new labels only"])
        self.labels_merged_or_seperated.setMinimumHeight(26)
        self.labels_merged_or_seperated.setMinimumWidth(260)
        self.labels_merged_or_seperated.setMaximumWidth(260)

        self.naming_label = QLabel("Name output images : ")

        self.naming_combobox = QComboBox()
        self.naming_combobox.addItems(["by indexing", "randomly"])
        self.naming_combobox.setMinimumHeight(26)
        self.naming_combobox.setMinimumWidth(160)
        self.naming_combobox.currentTextChanged.connect(self.naming_index_start_switch)
        self.naming_combobox.setMinimumHeight(26)

        self.naming_optional_part = QLineEdit()
        self.naming_optional_part.setMinimumWidth(145)
        self.naming_optional_part.setMaximumWidth(145)
        self.naming_optional_part.setMinimumHeight(26)
        self.naming_optional_part.setPlaceholderText("optional part")

        self.naming_index_start = QLineEdit()
        self.naming_index_start.setMinimumWidth(145)
        self.naming_index_start.setMaximumWidth(145)
        self.naming_index_start.setMinimumHeight(26)
        self.naming_index_start.setPlaceholderText("starting index")

        self.saving_format = QComboBox()
        self.saving_format.addItems([".jpg", ".png", ".jpeg"])
        self.saving_format.setMinimumHeight(26)

        self.augmentation_all_images_button = QPushButton("Start")
        self.augmentation_all_images_button.setDisabled(True)
        self.augmentation_all_images_button.clicked.connect(self.apply_augmentation_to_all_images)

        self.pbar = QProgressBar()
        self.pbar.setMinimumWidth(670)
        self.pbar.setMaximumWidth(670)
        self.pbar.setMinimumHeight(30)
        self.pbar.setMaximumHeight(30)
        ## ///augmenteatin to all images
        
        # ///other widgets - bottom left region
        # /// left layout
        # right layout
        self.image_name_label = QLabel("original image")
        self.image_name_label.setFont(QFont("Times", 9, QFont.Bold))

        self.image_switch_label_start = QLabel("Switch to ")
        self.original_image_button = QPushButton("original")
        self.original_image_button.setDisabled(True)
        self.original_image_button.clicked.connect(self.open_selected_image)
        self.augmented_image_button = QPushButton("augmented")
        self.augmented_image_button.setDisabled(True)
        self.augmented_image_button.clicked.connect(self.open_selected_augmented_image)
        self.image_switch_label_end = QLabel("image")
        
        self.image_label = QLabel("image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("image")

        self.scroll_area_image = QScrollArea()
        self.scroll_area_image.setMinimumHeight(500)
        self.scroll_area_image.setMinimumWidth(500)
        self.scroll_area_image.setMaximumHeight(750)
        self.scroll_area_image.setMaximumWidth(832)
        self.scroll_area_image.setWidget(self.image_label)
        self.scroll_area_image.setWidgetResizable(True)
        self.scroll_area_image.mouseMoveEvent = self.move_image

        self.original_informations_label = QLabel()
        self.original_informations_label.setFont(QFont("Times", 10))
        self.augment_informations_label = QLabel()
        self.augment_informations_label.setFont(QFont("Times", 10))
        # ////augmented image region -- right layout

    def get_values_from_sliders(self):

        self.brightness_interval = [ i - 100 for i in self.interval_brightness_value.value() ]
        self.contrast_interval = [ (i + 4) / 10 for i in self.interval_contrast_value.value() ]
        self.rotate_interval = [ i - 10 for i in self.interval_rotate_value.value() ]
        self.shearX_interval = [ (i - 20) / 100 for i in self.interval_shearX_value.value() ] 
        self.shearY_interval = [ (i - 20) / 100 for i in self.interval_shearY_value.value() ] 
        self.shiftX_interval = [ (i - 20) / 100 for i in self.interval_shiftX_value.value() ] 
        self.shiftY_interval = [ (i - 20) / 100 for i in self.interval_shiftY_value.value() ] 
        self.flip = self.flip_bool.value()
        self.brightness_possibility = self.possibility_brightness_value.value() / 100
        self.contrast_possibility = self.possibility_contrast_value.value() / 100
        self.rotate_possibility = self.possibility_rotate_value.value() / 100
        self.shearX_possibility = self.possibility_shearX_value.value() / 100
        self.shearY_possibility = self.possibility_shearY_value.value() / 100
        self.shiftX_possibility = self.possibility_shiftX_value.value() / 100
        self.shiftY_possibility = self.possibility_shiftY_value.value() / 100
        self.flip_possibility = self.possibility_flip.value() / 100


    def update_interval(self):

        self.get_values_from_sliders()

        self.interval_brightness_value_label.clear()
        self.interval_contrast_value_label.clear()
        self.interval_rotate_value_label.clear()
        self.interval_shearX_value_label.clear()
        self.interval_shearY_value_label.clear()
        self.interval_shiftX_value_label.clear()
        self.interval_shiftY_value_label.clear()
        self.flip_label.clear()
        self.possibility_brightness_value_label.clear()
        self.possibility_contrast_value_label.clear()
        self.possibility_rotate_value_label.clear()
        self.possibility_shearX_value_label.clear()
        self.possibility_shearY_value_label.clear()
        self.possibility_shiftX_value_label.clear()
        self.possibility_shiftY_value_label.clear()
        self.possibility_flip_label.clear()

        self.interval_brightness_value_label.setText("internal of brightness value : {} - {}".format(self.brightness_interval[0], self.brightness_interval[1]))
        self.interval_contrast_value_label.setText("internal of contrast value : {} - {}".format(self.contrast_interval[0], self.contrast_interval[1]))
        self.interval_rotate_value_label.setText("internal of rotation angle : {}° - {}°".format(self.rotate_interval[0], self.rotate_interval[1]))
        self.interval_shearX_value_label.setText("interval of shearing X value :  {} - {}".format(self.shearX_interval[0], self.shearX_interval[1]))
        self.interval_shearY_value_label.setText("interval of shearing Y value :  {} - {}".format(self.shearY_interval[0], self.shearY_interval[1]))
        self.interval_shiftX_value_label.setText("interval of shifting X value :  {} - {}".format(self.shiftX_interval[0], self.shiftX_interval[1]))
        self.interval_shiftY_value_label.setText("interval of shifting Y value :  {} - {}".format(self.shiftY_interval[0], self.shiftY_interval[1]))
        self.flip_label.setText("Flip : {}".format("(Apply flip)" if self.flip == 1 else "(Do not Apply Flip)"))
        self.possibility_brightness_value_label.setText("possibility of brightness value : {}".format(int(self.brightness_possibility*100)))
        self.possibility_contrast_value_label.setText("possibility of contrast value : {}".format(int(self.contrast_possibility*100)))
        self.possibility_rotate_value_label.setText("possibility of rotation value : {}".format(int(self.rotate_possibility*100)))
        self.possibility_shearX_value_label.setText("possibility of shearing X value : {}".format(int(self.shearX_possibility*100)))
        self.possibility_shearY_value_label.setText("possibility of shearing Y value : {}".format(int(self.shearY_possibility*100)))
        self.possibility_shiftX_value_label.setText("possibility of shifting X value : {}".format(int(self.shiftX_possibility*100)))
        self.possibility_shiftY_value_label.setText("possibility of shifting Y value : {}".format(int(self.shiftY_possibility*100)))
        self.possibility_flip_label.setText("possibility of Flip : {}".format(int(self.flip_possibility*100)))
        

    def apply_augmentation_to_selected_image(self):
        
        self.get_values_from_sliders()

        image = Image.open("scripts/original_image.jpg")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = self.image_name

        self.applied_adjust = ""
        if np.random.rand() < self.brightness_possibility:
            image, random_parameter = img_random_brightness(image, self.brightness_interval)
            self.applied_adjust = self.applied_adjust + "Brightness adjustment : " + str(np.round(random_parameter, 3)) + "\n"
        if np.random.rand() < self.contrast_possibility:
            image, random_parameter = img_random_contrasts(image, self.contrast_interval)
            self.applied_adjust = self.applied_adjust + "Contrast adjustment : " + str(np.round(random_parameter, 3)) + "\n"

        control = self.bounding_boxes_checkbox.isChecked()
        if control and self.label_control:
            try:
                coor = yolo_points_to_cv2_points(filename, self.DataFrame)
            except:
                QMessageBox.warning(self, "Unmatched labels", "There is not any label for this image !!!")
                self.bounding_boxes_checkbox.setChecked(False)
                return
            new_coor = coor
            if np.random.rand() < self.flip_possibility and self.flip:
                image, new_coor = img_random_flip(image, new_coor, True)
                self.applied_adjust = self.applied_adjust + "Flip operation \n"
            if np.random.rand() < self.rotate_possibility:
                image, new_coor, random_parameter = img_random_rotate(image, self.rotate_interval, new_coor, True)
                self.applied_adjust = self.applied_adjust + "Rotation : " + str(np.round(random_parameter, 3)) + "°\n"
            if np.random.rand() < self.shearX_possibility:
                image, new_coor, random_parameter = shear(image, self.shearX_interval, "X", new_coor, True)
                self.applied_adjust = self.applied_adjust + "Shear in X direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shearY_possibility:
                image, new_coor, random_parameter = shear(image, self.shearY_interval, "Y", new_coor, True)
                self.applied_adjust = self.applied_adjust + "Shear in Y direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shiftX_possibility:
                image, new_coor, random_parameter = translation(image, self.shiftX_interval, "X", new_coor, True)
                self.applied_adjust = self.applied_adjust + "Translation in X direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shiftY_possibility:
                image, new_coor, random_parameter = translation(image, self.shiftY_interval, "Y", new_coor, True)
                self.applied_adjust = self.applied_adjust + "Translation in Y direction : " + str(np.round(random_parameter * 100, 3)) + "%"

            cv2.imwrite("scripts/augmented_image.jpg", image)
            labels = give_labels(self.image_name, self.DataFrame)
            labels, new_coor = adjust(labels, new_coor)
            control2 = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, new_coor, labels, True)    

        else:
            self.bounding_boxes_checkbox.setChecked(False)
            if np.random.rand() < self.flip_possibility and self.flip:
                image = img_random_flip(image)
                self.applied_adjust = self.applied_adjust + "Flip operation \n"
            if np.random.rand() < self.rotate_possibility:
                image, random_parameter = img_random_rotate(image, self.rotate_interval)
                self.applied_adjust = self.applied_adjust + "Rotation : " + str(np.round(random_parameter, 3)) + "°\n"
            if np.random.rand() < self.shearX_possibility:
                image, random_parameter = shear(image, self.shearX_interval, "X")
                self.applied_adjust = self.applied_adjust + "Shear in X direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shearY_possibility:
                image, random_parameter = shear(image, self.shearY_interval, "Y")
                self.applied_adjust = self.applied_adjust + "Shear in Y direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shiftX_possibility:
                image, random_parameter = translation(image, self.shiftX_interval, "X")
                self.applied_adjust = self.applied_adjust + "Translation in X direction : " + str(np.round(random_parameter * 100, 3)) + "%\n"
            if np.random.rand() < self.shiftY_possibility:
                image, random_parameter = translation(image, self.shiftY_interval, "Y")
                self.applied_adjust = self.applied_adjust + "Translation in Y direction : " + str(np.round(random_parameter * 100, 3)) + "%"

            cv2.imwrite("scripts/augmented_image.jpg", image)
            control2 = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, on_aug_img=True)

        self.augment_informations_label.clear()
        self.augment_informations_label.setText(self.applied_adjust)
        if control2:
            self.image_label.clear()
            image = QImage("scripts/resized_augmented_image.jpg")
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.original_image_button.setEnabled(True)
            self.augmented_image_button.setDisabled(True)


    def naming_index_start_switch(self):
        if self.naming_combobox.currentText() == "randomly":
            self.naming_index_start.setDisabled(True)
            self.naming_optional_part.setDisabled(True)
        elif self.naming_combobox.currentText() == "by indexing":
            self.naming_index_start.setEnabled(True)
            self.naming_optional_part.setEnabled(True)
        else:
            print("wrong response : ", self.naming_combobox.currentText())


    def select_image_saving_folder(self):
        save_folder = QFileDialog.getExistingDirectory(self, caption='Select a folder')

        if save_folder:
            self.save_image_folder.clear()
            self.save_image_folder.setText(save_folder)


    def select_df_saving_folder(self):
        save_folder = QFileDialog.getExistingDirectory(self, caption='Select a folder')

        if save_folder:
            self.save_df_folder.clear()
            self.save_df_folder.setText(save_folder)


    def change_output_size_checkbox_function(self):
        if self.change_output_size_checkbox.isChecked():
            self.output_height.setEnabled(True)
            self.output_width.setEnabled(True)
        else:
            self.output_height.setDisabled(True)
            self.output_width.setDisabled(True)


    def save_parameters_as_json(self):

        params_dict = {
            "brightness_interval" : self.brightness_interval,
            "contrast_interval" : self.contrast_interval,
            "rotate_interval" : self.rotate_interval,
            "shearX_interval" : self.shearX_interval,
            "shearY_interval" : self.shearY_interval,
            "shiftX_interval" : self.shiftX_interval,
            "shiftY_interval" : self.shiftY_interval,
            "flip" : self.flip,
            "brightness_possibility" : self.brightness_possibility,
            "contrast_possibility" : self.contrast_possibility,
            "rotate_possibility" : self.rotate_possibility,
            "shearX_possibility" : self.shearX_possibility,
            "shearY_possibility" : self.shearY_possibility,
            "shiftX_possibility" : self.shiftX_possibility,
            "shiftY_possibility" : self.shiftY_possibility,
            "flip_possibility" : self.flip_possibility,   
            "image_saving_folder" : self.save_image_folder.text(),
            "label_saving_folder" : self.save_df_folder.text(),
            "labels_merged_or_seperated" : self.labels_merged_or_seperated.currentText(),
            "naming_method" : self.naming_combobox.currentText(),
            "optional_part" : self.naming_optional_part.text(),
            "saving_format" : self.saving_format.currentText(),
            "image_paths" : self.filenames,
            "control" : self.bounding_boxes_checkbox.isChecked(),
            "label_control" : self.label_control
        }

        if self.bounding_boxes_checkbox.isChecked() and self.label_control:
            params_dict["csv_file"] = self.csv_file

        if self.change_output_size_checkbox.isChecked():
            try:
                output_width = int(self.output_width.text())
                output_height = int(self.output_height.text())
                params_dict["change_output_image_size"] = True
                params_dict["output_width"] = output_width
                params_dict["output_height"] = output_height
            except:
                QMessageBox.warning(self, "Not a number", "Please give output image width and height values as numbers")
        else:
            params_dict["change_output_image_size"] = False
            
        if self.naming_combobox.currentText() == "by indexing":
            params_dict["starting_number"] = int(self.naming_index_start.text())
        else:
            params_dict["starting_number"] = 1
            
        with open('scripts/parameters.json', 'w') as json_file:
           json.dump(params_dict, json_file)


    def apply_augmentation_to_all_images(self):

        image_saving_folder = self.save_image_folder.text()
        label_saving_folder = self.save_df_folder.text()
        naming_method = self.naming_combobox.currentText()
        starting_number = self.naming_index_start.text()
        
        if os.path.exists(image_saving_folder) == False:
            QMessageBox.critical(self, "path error !!", "given image saving path does not exist or broken")
            return
        
        if naming_method == "by indexing":
            try:
                starting_number = int(starting_number)
            except:
                QMessageBox.warning(self, "Not a number", "Please give starting number value for naming images")
                self.augmentation_all_images_button.setEnabled(True)
                return

        if self.bounding_boxes_checkbox.isChecked() and self.label_control:
            if os.path.exists(label_saving_folder) == False:
                QMessageBox.critical(self, "path error !!", "given label saving path does not exist or broken")
                return
            else:
                response = QMessageBox.question(self, 'information', "Do you want to start to apply augmentations to all images and their bounding boxes?", QMessageBox.Yes | QMessageBox.No)

        elif self.bounding_boxes_checkbox.isChecked() and self.label_control == 0:
            response = QMessageBox.question(self, 'warning', "There is not any label for these images, Do you want to start apply augmentations to all images without appliying bounding boxes?", QMessageBox.Yes | QMessageBox.No)

        elif self.bounding_boxes_checkbox.isChecked() == 0 and self.label_control:
            response = QMessageBox.question(self, 'warning', "Do you want to start apply augmentations to all images without appliying bounding boxes?", QMessageBox.Yes | QMessageBox.No)

        elif self.bounding_boxes_checkbox.isChecked() == 0 and self.label_control  == 0:
            response = QMessageBox.question(self, 'information', "Do you want to start apply augmentations to all images?", QMessageBox.Yes | QMessageBox.No)


        if response == QMessageBox.Yes:

            def open_buttons():
                self.augmentation_all_images_button.setEnabled(True)
                self.augmentation_selected_image_button.setEnabled(True)
                QMessageBox.information(self, "Success", "Augmentations with given paramaters were succesfully applied to all images")

            self.get_values_from_sliders()
            self.save_parameters_as_json()

            self.thread = QThread()

            self.worker = Worker()

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)

            self.thread.start()

            self.augmentation_all_images_button.setEnabled(False)
            self.augmentation_selected_image_button.setEnabled(False)
            self.thread.finished.connect(open_buttons)
            self.thread.finished.connect(lambda: self.pbar.setValue(0))

            
    def reportProgress(self, n):
        
        self.pbar.setValue(n)
     
     
    def show_table_function(self) :
        
        self.showcsvapp = ShowCSVApp(self.csv_file)
        self.showcsvapp.show()
       
        
    def PopUp_table_window(self):
        
        open_csv = self.open_csv()
        self.opencsvapp = OpenCSVApp(open_csv, self.csv_file)
        self.opencsvapp.show()
        
    
    def open_csv(self):
        
        self.DataFrame = preprocess_image_labels(self.csv_file)

        self.label_control = 1
        self.bounding_boxes_checkbox.setChecked(True)
        self.show_table_button.setEnabled(True)

        if self.image_names != 0:
            self.open_selected_image("ignore")
            self.augmented_image_button.setDisabled(True)


    def get_csv_path(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.csv_file, _ = QFileDialog.getOpenFileName(self, "Open labels", "", "csv (*.csv *.txt);;All Files (*)", options=options)

        if self.csv_file:
            self.PopUp_table_window()


    def open_image(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filenames, _ = QFileDialog.getOpenFileNames(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)

        if filenames:
            
            self.filenames = filenames
            self.image_name = self.filenames[0]
            self.dirname = os.path.dirname(self.image_name)

            self.image_name_label.clear()
            self.image_name_label.setText(self.image_name)

            self.image_names = QListWidget()
            self.image_names.mouseDoubleClickEvent = self.open_selected_image

            self.items_in_listwidget = []
            for img in self.filenames:
                item = QListWidgetItem(os.path.basename(img))
                self.image_names.addItem(item)
                self.items_in_listwidget.append(item)
            self.image_names.setCurrentItem(self.items_in_listwidget[0])

            self.Layouts()

            if self.label_control == 1:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
            else:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)
                
            if control:
                self.image_label.clear()
                image = QImage("scripts/resized_original_image.jpg")
                self.image_label.setPixmap(QPixmap.fromImage(image))

            self.original_informations_label.clear()
            self.original_informations_label.setText(info)
            
            self.previous_image_button.setEnabled(True)
            self.size_button_smaller.setEnabled(True)
            self.size_button_bigger.setEnabled(True)
            self.next_image_button.setEnabled(True)
            self.augmentation_selected_image_button.setEnabled(True)
            self.augmentation_all_images_button.setEnabled(True)


    def open_selected_image(self, event):

        try:
            self.image_name = self.image_names.currentItem().text()
            self.image_name = self.dirname + "/" + self.image_name
        except:
            return

        self.image_name_label.clear()
        self.image_name_label.setText(self.image_name)
        
        if self.label_control == 1:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
        else:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)

        if control:
            self.image_label.clear()
            image = QImage("scripts/resized_original_image.jpg")
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.original_image_button.setDisabled(True)
            
        if type(event) == bool:
            self.augmented_image_button.setEnabled(True)
        elif event == "ignore":
            self.augment_informations_label.clear()
        elif event.button() == Qt.LeftButton:
            self.augmented_image_button.setDisabled(True)
            self.augment_informations_label.clear()
        else:
            print("unknown event")

        self.original_informations_label.clear()
        self.original_informations_label.setText(info)


    def open_selected_augmented_image(self):

        self.image_label.clear()
        image = QImage("scripts/resized_augmented_image.jpg")
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.original_image_button.setEnabled(True)
        self.augmented_image_button.setDisabled(True)
        self.augment_informations_label.clear()
        self.augment_informations_label.setText(self.applied_adjust)


    def get_previous_image(self):

        current_image = self.image_name
        images = self.filenames

        current_image_index = images.index(current_image) - 1
        self.image_name = images[current_image_index]
        self.image_names.setCurrentItem(self.items_in_listwidget[current_image_index])
        
        if self.label_control == 1:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
        else:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)

        if control:
            self.image_label.clear()
            image = QImage("scripts/resized_original_image.jpg")
            self.image_label.setPixmap(QPixmap.fromImage(image))
        
        self.original_image_button.setDisabled(True)
        self.augmented_image_button.setDisabled(True)
        self.augment_informations_label.clear()
        self.original_informations_label.clear()
        self.original_informations_label.setText(info)

        self.image_name_label.clear()
        self.image_name_label.setText(self.image_name)


    def image_get_smaller(self):

        if self.image_name:
            self.original_image_width -= 64
            self.original_image_height -= 64

            if self.label_control == 1:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
            else:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)
    
            if control:
                self.image_label.clear()
                image = QImage("scripts/resized_original_image.jpg")
                self.image_label.setPixmap(QPixmap.fromImage(image))
                self.original_image_button.setDisabled(True)
            
            self.original_informations_label.clear()
            self.original_informations_label.setText(info)


    def image_get_bigger(self):

        if self.image_name:
            self.original_image_width += 64
            self.original_image_height += 64

            if self.label_control == 1:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
            else:
                control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)

            if control:
                self.image_label.clear()
                image = QImage("scripts/resized_original_image.jpg")
                self.image_label.setPixmap(QPixmap.fromImage(image))
                self.original_image_button.setDisabled(True)

            self.original_informations_label.clear()
            self.original_informations_label.setText(info)


    def get_next_image(self):

        current_image = self.image_name
        images = self.filenames

        current_image_index = images.index(current_image) + 1
        if current_image_index < len(images):
            self.image_name = images[current_image_index]
        else:
            current_image_index = 0
            self.image_name = images[current_image_index]

        self.image_names.setCurrentItem(self.items_in_listwidget[current_image_index])

        if self.label_control == 1:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height, self.DataFrame)
        else:
            control, info = Give_Pixmap(self.image_name, self.original_image_width, self.original_image_height)

        if control:
            self.image_label.clear()
            image = QImage("scripts/resized_original_image.jpg")
            self.image_label.setPixmap(QPixmap.fromImage(image))

        self.original_image_button.setDisabled(True)
        self.augmented_image_button.setDisabled(True)
        self.augment_informations_label.clear()
        self.original_informations_label.clear()
        self.original_informations_label.setText(info)

        self.image_name_label.clear()
        self.image_name_label.setText(self.image_name)
        
        
    def move_image(self, event):
        
        x = event.pos().x()
        y = event.pos().y() 

        y_max = self.scroll_area_image.verticalScrollBar().maximum()
        x_max = self.scroll_area_image.horizontalScrollBar().maximum()

        self.scroll_area_image.verticalScrollBar().setValue(int(y_max*1.2) - (y))
        self.scroll_area_image.horizontalScrollBar().setValue(int(x_max*1.2) - (x))



    def Layouts(self):

        main_widget = QGroupBox()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Left layout
        left_horizontal_layout = QHBoxLayout()
        ## imported images
        imported_images_layout = QVBoxLayout()

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.previous_image_button)
        buttons_layout.addWidget(self.size_button_smaller)
        buttons_layout.addWidget(self.size_button_bigger)
        buttons_layout.addWidget(self.next_image_button) 
        buttons_layout.addStretch()

        images_main_groupBox = QGroupBox("original images")
        images_main_groupBox.setMinimumHeight(443)
        images_main_groupBox.setMinimumWidth(200)

        List_widget_box = QHBoxLayout()
        if self.image_names:
            List_widget_box.addWidget(self.image_names)
        images_main_groupBox.setLayout(List_widget_box)

        List_widget_groupBox_layout = QHBoxLayout()
        List_widget_groupBox_layout.addWidget(images_main_groupBox)

        imported_images_layout.addLayout(buttons_layout)
        imported_images_layout.addLayout(List_widget_groupBox_layout)
        ## ///imported images
        ## adjustments
        sliders_horizontal_layout = QHBoxLayout()
        sliders_vertical_layout = QVBoxLayout()

        interval_sliders_layout = QVBoxLayout()

        interval_sliders_layout.addWidget(self.interval_brightness_value_label)
        interval_sliders_layout.addWidget(self.interval_brightness_value)
        interval_sliders_layout.addWidget(self.interval_contrast_value_label)
        interval_sliders_layout.addWidget(self.interval_contrast_value)
        interval_sliders_layout.addWidget(self.interval_rotate_value_label)
        interval_sliders_layout.addWidget(self.interval_rotate_value)
        interval_sliders_layout.addWidget(self.interval_shearX_value_label)
        interval_sliders_layout.addWidget(self.interval_shearX_value)
        interval_sliders_layout.addWidget(self.interval_shearY_value_label)
        interval_sliders_layout.addWidget(self.interval_shearY_value)
        interval_sliders_layout.addWidget(self.interval_shiftX_value_label)
        interval_sliders_layout.addWidget(self.interval_shiftX_value)
        interval_sliders_layout.addWidget(self.interval_shiftY_value_label)
        interval_sliders_layout.addWidget(self.interval_shiftY_value)
        interval_sliders_layout.addWidget(self.flip_label)
        interval_sliders_layout.addWidget(self.flip_bool)

        possibilities_sliders_layout = QVBoxLayout()

        possibilities_sliders_layout.addWidget(self.possibility_brightness_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_brightness_value)
        possibilities_sliders_layout.addWidget(self.possibility_contrast_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_contrast_value)
        possibilities_sliders_layout.addWidget(self.possibility_rotate_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_rotate_value)
        possibilities_sliders_layout.addWidget(self.possibility_shearX_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_shearX_value)
        possibilities_sliders_layout.addWidget(self.possibility_shearY_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_shearY_value)
        possibilities_sliders_layout.addWidget(self.possibility_shiftX_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_shiftX_value)
        possibilities_sliders_layout.addWidget(self.possibility_shiftY_value_label)
        possibilities_sliders_layout.addWidget(self.possibility_shiftY_value)
        possibilities_sliders_layout.addWidget(self.possibility_flip_label)
        possibilities_sliders_layout.addWidget(self.possibility_flip)

        interval_sliders_GroupBox = QGroupBox("Random values intervals")
        interval_sliders_GroupBox.setMinimumWidth(275)
        interval_sliders_GroupBox.setLayout(interval_sliders_layout)

        possibilities_sliders_Groupbox = QGroupBox("Possibilities of Applying Augmentation")
        possibilities_sliders_Groupbox.setMinimumWidth(275)
        possibilities_sliders_Groupbox.setLayout(possibilities_sliders_layout)

        sliders_horizontal_layout.addWidget(interval_sliders_GroupBox)
        sliders_horizontal_layout.addWidget(possibilities_sliders_Groupbox)
        
        sliders_vertical_layout.addLayout(sliders_horizontal_layout)
        
        left_horizontal_layout.addLayout(imported_images_layout)
        left_horizontal_layout.addLayout(sliders_vertical_layout)

        ## // adjsutments
        ## other widgets -- left bottom region
        left_bottom_layout = QVBoxLayout()
        checkbox_hbox = QHBoxLayout()
        checkbox_hbox.addStretch()
        checkbox_hbox.addWidget(self.bounding_boxes_checkbox)
        checkbox_hbox.addStretch()
        left_bottom_layout.addLayout(checkbox_hbox)

        augment_selected_image_groupbox = QGroupBox("Augmentation on Selected Image")

        augment_selected_image_groupbox_hbox = QHBoxLayout()
        augment_selected_image_groupbox_hbox_left = QHBoxLayout()
        augment_selected_image_groupbox_hbox_left.addStretch()
        augment_selected_image_groupbox_hbox_left.addWidget(self.augmentation_selected_image_label)
        augment_selected_image_groupbox_hbox_right = QHBoxLayout()
        augment_selected_image_groupbox_hbox_right.addWidget(self.augmentation_selected_image_button)
        augment_selected_image_groupbox_hbox_right.addWidget(self.show_table_button)
        augment_selected_image_groupbox_hbox_right.addStretch()

        augment_selected_image_groupbox_hbox.addLayout(augment_selected_image_groupbox_hbox_left, 50)
        augment_selected_image_groupbox_hbox.addLayout(augment_selected_image_groupbox_hbox_right, 50)
        augment_selected_image_groupbox_hbox.addStretch()

        augment_selected_image_groupbox.setLayout(augment_selected_image_groupbox_hbox)

        augment_all_images_groupbox = QGroupBox("Augmentation on All Images")

        augment_all_images_groupbox_hbox_1 = QHBoxLayout()
        augment_all_images_groupbox_hbox_1.addStretch()
        augment_all_images_groupbox_hbox_1.addWidget(self.save_image_folder)
        augment_all_images_groupbox_hbox_1.addWidget(self.save_image_folder_button)
        augment_all_images_groupbox_hbox_1.addWidget(self.change_output_size_checkbox)
        augment_all_images_groupbox_hbox_1.addWidget(self.output_width)
        augment_all_images_groupbox_hbox_1.addWidget(self.output_height)
        augment_all_images_groupbox_hbox_1.addStretch()

        augment_all_images_groupbox_hbox_2 = QHBoxLayout()
        augment_all_images_groupbox_hbox_2.addStretch()
        augment_all_images_groupbox_hbox_2.addWidget(self.save_df_folder)
        augment_all_images_groupbox_hbox_2.addWidget(self.save_df_folder_button)
        augment_all_images_groupbox_hbox_2.addWidget(self.labels_merged_or_seperated)
        augment_all_images_groupbox_hbox_2.addStretch()

        augment_all_images_groupbox_hbox_3 = QHBoxLayout()
        augment_all_images_groupbox_hbox_3.addStretch()
        augment_all_images_groupbox_hbox_3.addWidget(self.naming_label)
        augment_all_images_groupbox_hbox_3.addWidget(self.naming_combobox)
        augment_all_images_groupbox_hbox_3.addWidget(self.naming_optional_part)
        augment_all_images_groupbox_hbox_3.addWidget(self.naming_index_start)
        augment_all_images_groupbox_hbox_3.addWidget(self.saving_format)
        augment_all_images_groupbox_hbox_3.addStretch()

        augment_all_images_groupbox_hbox_4 = QHBoxLayout()
        augment_all_images_groupbox_hbox_4.addStretch()
        augment_all_images_groupbox_hbox_4.addWidget(self.augmentation_all_images_button)
        augment_all_images_groupbox_hbox_4.addStretch()

        augment_all_images_groupbox_hbox_5 = QHBoxLayout()
        augment_all_images_groupbox_hbox_5.addStretch()
        augment_all_images_groupbox_hbox_5.addWidget(self.pbar)
        augment_all_images_groupbox_hbox_5.addStretch()

        augment_all_images_vbox = QVBoxLayout()
        augment_all_images_vbox.addLayout(augment_all_images_groupbox_hbox_1)
        augment_all_images_vbox.addLayout(augment_all_images_groupbox_hbox_2)
        augment_all_images_vbox.addLayout(augment_all_images_groupbox_hbox_3)
        augment_all_images_vbox.addLayout(augment_all_images_groupbox_hbox_4)
        augment_all_images_vbox.addLayout(augment_all_images_groupbox_hbox_5)

        augment_all_images_groupbox.setLayout(augment_all_images_vbox)

        left_bottom_layout.addWidget(augment_selected_image_groupbox)
        left_bottom_layout.addWidget(augment_all_images_groupbox)

        ## //other widgets -- left bottom region
        left_layout.addLayout(left_horizontal_layout)
        left_layout.addLayout(left_bottom_layout)

        left_GroupBox = QGroupBox("adjustments")
        left_GroupBox.setLayout(left_layout)
        # ///Left layout
        # Right Layout
        ## images
        opened_image_layout = QVBoxLayout()
        opened_image_layout.addWidget(self.image_name_label)

        switch_buttons_layout = QHBoxLayout()
        switch_buttons_layout.addStretch()
        switch_buttons_layout.addWidget(self.image_switch_label_start)
        switch_buttons_layout.addWidget(self.original_image_button)
        switch_buttons_layout.addWidget(self.augmented_image_button)
        switch_buttons_layout.addWidget(self.image_switch_label_end)
        switch_buttons_layout.addStretch()

        information_hbox_layout = QHBoxLayout()
        original_information_groupBox = QGroupBox("Original image metadata")
        original_information_groupBox.setMinimumHeight(225)
        original_information_groupBox.setMaximumHeight(225)
        original_information_vbox = QVBoxLayout()
        original_information_vbox.addWidget(self.original_informations_label)
        original_information_vbox.addStretch()
        original_information_groupBox.setLayout(original_information_vbox)

        augmented_information_groupBox = QGroupBox("Applied augmentations")
        augmented_information_groupBox.setMinimumHeight(225)
        augmented_information_groupBox.setMaximumHeight(225)
        augmented_information_vbox = QVBoxLayout()
        augmented_information_vbox.addWidget(self.augment_informations_label)
        augmented_information_vbox.addStretch()
        augmented_information_groupBox.setLayout(augmented_information_vbox)

        information_hbox_layout.addWidget(original_information_groupBox)
        information_hbox_layout.addWidget(augmented_information_groupBox)

        opened_image_layout.addLayout(switch_buttons_layout)
        opened_image_layout.addWidget(self.scroll_area_image)
        opened_image_layout.addLayout(information_hbox_layout)
        #opened_image_layout.addStretch()

        ## ///images
        right_layout = QVBoxLayout()
        right_layout.addLayout(opened_image_layout)
        right_GroupBox = QGroupBox("İmages")
        right_GroupBox.setLayout(right_layout)
        # ///Right Layout
        main_layout.addWidget(left_GroupBox, 50)
        main_layout.addWidget(right_GroupBox, 50)

        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
