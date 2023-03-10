import pandas as pd
from PyQt5.QtWidgets import QWidget, QTableWidget,QTableWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea, QMessageBox, QLineEdit, QLabel
from PyQt5.QtGui import QIcon


class OpenCSVApp(QWidget):
    
    def __init__(self, coming_funtion, file_name):
        super().__init__()
        self.coming_funtion = coming_funtion
        self.file_name = file_name

        self.setWindowTitle("Table")
        self.setGeometry(200, 200, 100, 100)
        self.setWindowIcon(QIcon("extras/icon.ico"))
         
        self.filenames_column_label = QLabel("         Select filenames column : ")
        self.filenames_column_input = QLineEdit()
        self.filenames_column_button = QPushButton("Select")
        self.filenames_column_button.clicked.connect(self.filenames_column_button_funtion)
        
        self.class_column_label = QLabel("     Select class names column : ")
        self.class_column_input = QLineEdit()
        self.class_column_button = QPushButton("Select")
        self.class_column_button.clicked.connect(self.class_column_button_function)
        
        self.bounding_box_column_label = QLabel("Select bounding boxes column : ")
        self.bounding_box_column_input = QLineEdit()
        self.bounding_box_column_button = QPushButton("Select")
        self.bounding_box_column_button.clicked.connect(self.bounding_box_column_button_function)

        self.table = QTableWidget(self)

        self.button = QPushButton("Ok")
        self.button.clicked.connect(self.button_funtion)
        self.button.setMinimumWidth(200)
        self.button.setMaximumWidth(200)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setMinimumHeight(500)
        self.scroll_area.setMinimumWidth(750)
        self.scroll_area.setWidget(self.table)
        self.scroll_area.setWidgetResizable(True)
        
        self.layouts()
        self.open_csv()
        

    def filenames_column_button_funtion(self):
        
        current_index = self.table.currentColumn()
        self.filenames_column_input.setText(self.column_names[current_index])
        
        
    def class_column_button_function(self):
        
        current_index = self.table.currentColumn()
        self.class_column_input.setText(self.column_names[current_index])
        
        
    def bounding_box_column_button_function(self):
        
        current_index = self.table.currentColumn()
        self.bounding_box_column_input.setText(self.column_names[current_index])
        

    def button_funtion(self):
        
        filenames = self.filenames_column_input.text()
        classes = self.class_column_input.text()
        bounding_boxes = self.bounding_box_column_input.text()
        
        if not filenames in self.column_names:
            QMessageBox.warning(self, "wrong column name", "given column name for 'filenames' is not in the table")
            return
        elif not classes in self.column_names:
            QMessageBox.warning(self, "wrong column name", "given column name for 'class names' is not in the table")
            return
        elif not bounding_boxes in self.column_names:
            QMessageBox.warning(self, "wrong column name", "given column name for 'bounding boxes' is not in the table")
            return
        else:
            self.button.setEnabled(True)
            
        self.df.rename(columns= {
            self.filenames_column_input.text() : "#filename",
            self.class_column_input.text() : "region_attributes",
            self.bounding_box_column_input.text() : "bounding_box"
        }, inplace=True)
        
        self.df.to_csv("extras/dataframe.csv", index=False)
        self.coming_funtion()
        self.destroy()


    def open_csv(self):
        
        if self.file_name:

            self.df = pd.read_csv(self.file_name)
            self.column_names = list(self.df.columns)
            try:
                self.filenames_column_input.setText(self.column_names[0])
                self.class_column_input.setText(self.column_names[1])
                self.bounding_box_column_input.setText(self.column_names[2])
            except:
                return

            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)

            self.table.setColumnCount(len(self.df.columns))
            self.table.setHorizontalHeaderLabels(self.df.columns)

            for i, row in self.df.iterrows():
                self.table.insertRow(self.table.rowCount())
                for j, cell in enumerate(row):
                    self.table.setItem(self.table.rowCount()-1, j, QTableWidgetItem(str(cell)))
      
    
    def layouts(self):
        main_layout = QVBoxLayout()
        
        filenames_column_layout = QHBoxLayout()
        filenames_column_layout.addStretch()
        filenames_column_layout.addWidget(self.filenames_column_label)
        filenames_column_layout.addWidget(self.filenames_column_input)
        filenames_column_layout.addWidget(self.filenames_column_button)
        filenames_column_layout.addStretch()
        
        class_column_layout = QHBoxLayout()
        class_column_layout.addStretch()
        class_column_layout.addWidget(self.class_column_label)
        class_column_layout.addWidget(self.class_column_input)
        class_column_layout.addWidget(self.class_column_button)
        class_column_layout.addStretch()
        
        bounding_boxes_column_layout = QHBoxLayout()
        bounding_boxes_column_layout.addStretch()
        bounding_boxes_column_layout.addWidget(self.bounding_box_column_label)
        bounding_boxes_column_layout.addWidget(self.bounding_box_column_input)
        bounding_boxes_column_layout.addWidget(self.bounding_box_column_button)
        bounding_boxes_column_layout.addStretch()
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.button)
        button_layout.addStretch()
        
        main_layout.addLayout(filenames_column_layout)
        main_layout.addLayout(class_column_layout)
        main_layout.addLayout(bounding_boxes_column_layout)
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
