import pandas as pd
from PyQt5.QtWidgets import QWidget, QTableWidget,QTableWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea, QHeaderView
from PyQt5.QtGui import QIcon


class ShowCSVApp(QWidget):
    
    def __init__(self, df):
        super().__init__()
        
        self.df = df

        self.setWindowTitle("Table")
        self.setGeometry(200, 200, 100, 200)
        self.setWindowIcon(QIcon("extras/icon.ico"))

        self.table = QTableWidget(self)

        self.button = QPushButton("Ok")
        self.button.clicked.connect(self.button_funtion)
        self.button.setMinimumWidth(200)
        self.button.setMaximumWidth(200)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setMinimumHeight(500)
        self.scroll_area.setMinimumWidth(770)
        self.scroll_area.setWidget(self.table)
        self.scroll_area.setWidgetResizable(True)
        
        self.layouts()
        self.open_csv()
        

    def button_funtion(self):

        self.destroy()


    def open_csv(self):

        self.table.clear()
        
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)
        
        for i, row in self.df.iterrows():
            self.table.insertRow(self.table.rowCount())
            for j, cell in enumerate(row):
                self.table.setItem(self.table.rowCount()-1, j, QTableWidgetItem(str(cell)))
                
        columns = list(self.df.columns)
        bounding_boxes_index = columns.index("bounding_box")
        self.table.setColumnWidth(bounding_boxes_index, 350)
      
    
    def layouts(self):
        
        main_layout = QVBoxLayout()
 
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.button)
        button_layout.addStretch()

        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)