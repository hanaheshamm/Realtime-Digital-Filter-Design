from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTableWidget, QTableWidgetItem, QMessageBox, \
    QFileDialog, QShortcut
from scipy.signal import freqz, lfilter, zpk2tf, filtfilt
# from gui import Ui_MainWindow
import numpy as np
import pyqtgraph as pg
from scipy.signal import freqz
import pandas as pd
import os
import scipy
import scipy.signal
import math
from numpy import *
from numpy.random import *
from scipy.signal import *
import pyqtgraph
import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import numpy as np
import os
from os import path
import sys
import numpy as np
import matplotlib.pyplot as plt

# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        self.viewports = [self.unitCircleGraph, self.magnitudeGraph, self.phaseGraph,
                          self.allPassGraph, self.InputGraph, self.filteredGraph, self.mouseInputGraph]
        self.plotTitles = ['Zero/Pole Plot', 'Magnitude Response', 'Phase Response', 'All Pass Response',
                           'Realtime Input', 'Filtered Output', 'Mouse Input']
        self.init_UI()

    def customize_plot(self, plot, title):
        plot.getPlotItem().showGrid(True, True)
        plot.getPlotItem().setTitle(title)
        plot.setMenuEnabled(False)

    def init_UI(self):
        # Customize the appearance of the plots using the new function
        for view, title in zip(self.viewports, self.plotTitles):
            self.customize_plot(view, title)

        self.zeros = []
        self.poles = []

        self.added = "Zeros"

        self.current_index = 0
        self.animation_speed = 1
        self.is_animation_running = False
        self.is_signal_ended = False

        self.toggle_rm = True

        self.x_last_selected, self.y_last_selected = None, None
        self.x_last_pair, self.y_last_pair = None, None

        self.point_moving = None

        self.point_selected = False

        self.pair_mode = False

        self.data_dict = {
            "Zeros": [],
            "Poles": [],
        }
        self.data_brush = {
            "Zeros": 'g',
            "Poles": 'r',
        }
        self.data_symbol = {
            "Zeros": 'o',
            "Poles": 'x',
        }
        self.data_plots = {
            "Data": self.InputGraph,
            "Data_modified": self.filteredGraph,
        }

        self.data = []
        self.data_modified = []

        self.all_pass_a = 1

        self.inputSignal = pg.PlotDataItem([], pen='b', width=2)
        self.filteredSignal = pg.PlotDataItem([], pen='b', width=2)
        self.InputGraph.addItem(self.inputSignal)
        self.filteredGraph.addItem(self.filteredSignal)

        self.allpass_en = False

        self.selected_point = None
        self.pair_selected = None
        self.selected_point_index = None

        self.data_opened = False

        self.point_type = "Zeros"

        self.moved_last_x = 0
        self.moved_last_y = 0

        self.mouse_loc_circle = None

        self.mouse_enable = False

        self.checked_coeffs = [0.0]

        self.frequencies = 0
        self.mag_response = 0
        self.phase_response = 0

        self.total_phase = 0

        self.colors = ['#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#800080', '#FF00FF',
                       '#FF1493', '#00FF7F', '#FFD700', '#FF6347', '#48D1CC', '#8A2BE2', '#20B2AA']

        self.btn_addZeros.clicked.connect(lambda: self.set_added("Zeros"))
        self.btn_addPoles.clicked.connect(lambda: self.set_added("Poles"))
        self.btn_RemoveZeros.clicked.connect(lambda: self.remove_points("Zeros"))
        self.btn_removePoles.clicked.connect(lambda: self.remove_points("Poles"))
        self.btn_removeAll.clicked.connect(self.remove_all)

        # Consolidate signal connections
        self.btn_addCoeff.clicked.connect(self.add_coefficient)
        self.btn_removeCoeff.clicked.connect(self.remove_coefficient)
        self.btn_openFile.clicked.connect(self.open_file)
        self.pair_mode_toggle.stateChanged.connect(self.toggle_pair_mode)
        self.all_pass_enable.stateChanged.connect(self.toggle_all_pass)

        # self.actionImport.triggered.connect(self.import_zeroes_poles)
        # self.actionExport.triggered.connect(self.export_zeros_poles)
        self.importButton.clicked.connect(self.import_zeroes_poles)
        self.exportButton.clicked.connect(self.export_zeros_poles)

        self.btn_play.clicked.connect(self.toggle_animation)

        self.mouse_en.stateChanged.connect(self.toggle_mouse_drawing)

        self.btn_addCoeff.clicked.connect(self.update_plot_allpass)
        self.btn_removeCoeff.clicked.connect(self.update_plot_allpass)
        self.table_coeff.itemChanged.connect(self.update_plot_allpass)

        self.speed_slider.valueChanged.connect(self.set_animation_speed)  # Set Animation speed when slide is changed

        self.btnClr.clicked.connect(self.clear_plots)

        self.move_clicked = False

        # Create circle ROIs to show the unit circle and an additional circle of radius 2
        self.roi_unitCircle = pg.CircleROI([-1, -1], [2, 2], pen=pg.mkPen('r', width=2), movable=False, resizable=False,
                                           rotatable=False)

        # Set the origin point to the center of the widget
        self.unitCircleGraph.setYRange(-1.1, 1.1, padding=0)
        self.unitCircleGraph.setXRange(-1.1, 1.1, padding=0)
        self.unitCircleGraph.setMouseEnabled(x=False, y=False)

        self.unitCircleGraph.addItem(self.roi_unitCircle)
        self.roi_unitCircle.removeHandle(0)

        self.unitCircleGraph.scene().sigMouseClicked.connect(self.on_click)
        self.unitCircleGraph.scene().sigMouseMoved.connect(self.drag_point)

        self.add_radioButton.setChecked(True)


        self.animation_timer = QTimer()
        #this calls update animation function each 30ms
        self.animation_timer.timeout.connect(self.update_animation)

        self.counter_max = 1
        self.counter_min = 0

        # Connect the sigMouseMoved signal to the on_mouse_move method in init_UI
        self.mouseInputGraph.scene().sigMouseMoved.connect(self.on_mouse_move)


    # Handling input signal real time filtering

    def set_play_button_state(self):
        state_dict = {
            True: "Stop",
            False: "Play",
        }
        self.btn_play.setText(state_dict[self.is_animation_running])

#label for speed slider
    def set_animation_speed(self):
        self.animation_speed = int(self.speed_slider.value())
        self.lbl_speed.setText(f"Speed: {self.animation_speed}")

    def play_animation(self):
        if self.is_signal_ended:
            print("Signal Ended")
            self.current_index = 0
            self.reset_viewport_range()
            self.is_signal_ended = False
        print("animation playing")
        self.animation_timer.start(30)
        self.is_animation_running = True
        self.set_play_button_state()

    def stop_animation(self):
        print("animation stopped")
        self.animation_timer.stop()
        self.is_animation_running = False
        self.set_play_button_state()

    def toggle_animation(self):
        self.is_animation_running = not self.is_animation_running

        if self.is_animation_running:
            self.filter_data()
            self.play_animation()
        else:
            self.stop_animation()

    def update_animation(self):
        x_min, x_max = self.viewports[4].viewRange()[0]
        self.inputSignal.setData(self.data[0:self.current_index])
        self.filteredSignal.setData(self.data_modified[0:self.current_index])

        if self.current_index > x_max:
            for viewport in [self.InputGraph, self.filteredGraph]:
                viewport.setLimits(xMax=self.current_index)
                viewport.getViewBox().translateBy(x=self.animation_speed)

        if self.current_index >= len(self.data) - 1:
            self.stop_animation()
            self.is_signal_ended = True

        self.current_index += self.animation_speed  # Convert the speed value to integer
        QApplication.processEvents()



    # Adding and Removing the All Pass Coefficients in the table
    def add_coefficient(self):
        # Create a QTableWidgetItem
        coeff_item = QTableWidgetItem(self.comboBox.currentText())
        coeff_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        coeff_item.setCheckState(Qt.CheckState.Checked)

        # Insert the item into the table widget
        # Inserts a new row at the end of the table
        self.table_coeff.insertRow(self.table_coeff.rowCount())

        self.table_coeff.setItem(self.table_coeff.rowCount() - 1, 0, coeff_item)

    # Removes the selected row from the table widget
    def remove_coefficient(self):
        self.table_coeff.removeRow(self.table_coeff.currentRow())




    # importing and exporting Zeros and Poles
    def export_zeros_poles(self):
        try:
            # Get the script directory and set the initial folder for saving files
            script_directory = os.path.dirname(os.path.abspath(__file__))
            initial_folder = os.path.join(script_directory, "Zeros-Poles")

            # Prompt the user to choose a file name and location for saving
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save File', initial_folder, "CSV files (*.csv)")
            path = file_name

            if path:
                # Ensure zeros and poles have the same length by padding with None
                max_len = max(len(self.data_dict["Zeros"]), len(self.data_dict["Poles"]))

                #Pads the zeros and poles lists with None to make them of equal length.
                zeros_padded = self.data_dict["Zeros"] + [None] * (max_len - len(self.data_dict["Zeros"]))
                poles_padded = self.data_dict["Poles"] + [None] * (max_len - len(self.data_dict["Poles"]))

                # Convert the data_dict to a DataFrame
                df = pd.DataFrame({
                    'Zeros_x': [zero.x() if zero else None for zero in zeros_padded],
                    'Zeros_y': [zero.y() if zero else None for zero in zeros_padded],
                    'Poles_x': [pole.x() if pole else None for pole in poles_padded],
                    'Poles_y': [pole.y() if pole else None for pole in poles_padded],
                })

                # Save the DataFrame to a CSV file
                df.to_csv(path, index=False)

        except Exception as e:
            print(f"Error: {e}")

    def import_zeroes_poles(self):
        try:
            # Clear existing data for Zeros and Poles
            self.data_dict["Zeros"] = []
            self.data_dict["Poles"] = []

            # Get the script directory and set the initial folder for opening files
            script_directory = os.path.dirname(os.path.abspath(__file__))
            initial_folder = os.path.join(script_directory, "Zeros-Poles")

            # Prompt the user to choose a file for opening
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', initial_folder, "CSV files (*.csv)")
            path = file_name

            if path:
                # Load data from the selected CSV file using pandas
                df = pd.read_csv(path)

                # Check if the columns have the same length
                if len(df['Zeros_x']) != len(df['Zeros_y']) or len(df['Poles_x']) != len(df['Poles_y']):
                    print("Error: Zeros and Poles arrays must be of the same length.")
                    return

                # Iterate through the loaded data and create Point instances for Zeros
                for x, y in zip(df['Zeros_x'], df['Zeros_y']):
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        point = pg.Point(x, y)
                        self.data_dict["Zeros"].append(point)

                # Iterate through the loaded data and create Point instances for Poles
                for x, y in zip(df['Poles_x'], df['Poles_y']):
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        point = pg.Point(x, y)
                        self.data_dict["Poles"].append(point)

                # Update the plot with the loaded data
                self.update_plot()

        except Exception as e:
            print(f"Error: {e}")



    # Opening signals
    def open_file(self):
        try:
            self.clear_plots()
            # Get the script directory and set the initial folder for file dialog
            script_directory = os.path.dirname(os.path.abspath(__file__))
            initial_folder = os.path.join(script_directory, "Data")

            # Display the file dialog to get the selected file
            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Open File', initial_folder, "CSV files (*.csv)"
            )

            # Check if the user canceled the file dialog
            if not file_name:
                return

            # Read the CSV file using pandas
            df = pd.read_csv(file_name)

            # Specify the column containing data
            data_col = 'Data'

            # Extract data from the specified column
            self.data = df[data_col].values

            # Create a copy of the original data for modification
            self.data_modified = self.data.copy()

            # Set the flag indicating that data has been opened
            self.data_opened = True

            # Reset the viewport range and update the plotted data
            self.reset_viewport_range()
            self.inputSignal.setData(self.data[0:self.current_index])
            self.filteredSignal.setData(self.data_modified[0:self.current_index])

            self.toggle_animation()

        except Exception as e:
            print(f"Error: {e}")





    # Handling clicking on unitCircleGraph

    def on_click(self, event):
        if event.button() == Qt.LeftButton:
            # if event.modifiers() == Qt.ControlModifier:
            if self.add_radioButton.isChecked():
                self.add_point(self.mouse_loc_circle, self.added)
                self.update_plot()
            # else:
            elif self.move_radioButton.isChecked():
                self.move_clicked = True
                self.move_point(self.mouse_loc_circle)

        elif event.button() == Qt.RightButton:
            if self.x_last_selected is not None and self.point_selected:
                self.unselect_moving_point()
            else:
                self.remove_point(self.mouse_loc_circle)

    def unselect_moving_point(self):
        #point_moving : is the old point i want to move (i will delete it and add it to a new location)
        #point: is the new location of the moved point
        self.remove_point(self.point_moving)
        point = pg.Point(self.x_last_selected, self.y_last_selected)
        self.add_point(point, self.point_type)
        if self.pair_selected:
            self.remove_point(self.point_moving_pair)
            point = pg.Point(self.x_last_selected, -self.y_last_selected)
            self.add_point(point, self.point_type)
        self.update_plot()
        #to indicate that no point is currently selected or being moved.
        self.x_last_selected, self.y_last_selected, self.point_selected, self.pair_selected, self.move_clicked = None, None, False, False, False



    # Moving point

    def move_point(self, pos_data):
        if self.x_last_selected is not None and self.point_selected:
            self.x_last_selected, self.y_last_selected, self.point_selected, self.pair_selected, self.move_clicked = None, None, False, False, False

        else:
            for dict_key in ["Zeros", "Poles"]:
                self.move_point_from_list(dict_key, pos_data)

    def move_point_from_list(self, dict_key, mouse_location):
        points_list = self.data_dict[dict_key].copy()
        for point in points_list:
            #points_list is a list of tupples of cordinates of zeroes or poles
            #The np.allclose function is used with a tolerance (atol) to account for potential floating-point discrepancies.
            if np.allclose([point.x(), point.y()], [mouse_location.x(), mouse_location.y()], atol=0.03):
                #location of last zero or pole selected by mouse
                self.x_last_selected, self.y_last_selected = point.x(), point.y()
                point_pair = pg.Point(point.x(), -point.y())

                self.point_type = dict_key

                self.point_selected = True
                if point_pair in self.data_dict[self.point_type]:
                    self.pair_selected = True

                self.point_moving = pg.Point(self.x_last_selected, self.y_last_selected)
                self.point_moving_pair = pg.Point(self.x_last_selected, -self.y_last_selected)
                break

    def drag_point(self, pos):
        pos = self.unitCircleGraph.getViewBox().mapSceneToView(pos)

        self.mouse_loc_circle = pg.Point(pos.x(), pos.y())
        self.mouse_loc_circle_pair = pg.Point(pos.x(), -pos.y())

        if self.move_clicked and self.point_selected:
            #we will remove the last selected zero or pole
            self.remove_point(self.point_moving)
            # we will add a new zero or pole in the location of the mouse
            self.add_point(self.mouse_loc_circle, self.point_type)
            if self.pair_selected:
                self.remove_point(self.point_moving_pair)
                self.add_point(self.mouse_loc_circle_pair, self.point_type)

            self.update_plot()
            # the location of last selected zero or pole will be at the location of the mouse because we dragged the zero/pole to the
            # new location of the mouse
            self.point_moving = self.mouse_loc_circle
            self.point_moving_pair = self.mouse_loc_circle_pair



    # Adding point

    # def create_point(self, x, y):
    #     return pg.Point(x, y)

    def add_point(self, point, dict_key):
        # Assuming x and y are coordinates
        point = pg.Point(point.x(), point.y())

        if self.pair_mode:
            point_pair = pg.Point(point.x(), -point.y())
            self.add_points_to_dict(point, point_pair, dict_key)
        else:
            self.add_points_to_dict(point, None, dict_key)

    def add_points_to_dict(self, point, point_pair, dict_key):
        self.data_dict[dict_key].append(point)
        if point_pair is not None:
            self.data_dict[dict_key].append(point_pair)



    # Removing point

    def remove_point(self, point_data):
        for dict_name in ["Zeros", "Poles"]:
            if self.remove_point_from_list(self.data_dict[dict_name], point_data, atol_cof=0):
                break
            elif self.remove_point_from_list(self.data_dict[dict_name], point_data):
                break

    def remove_point_from_list(self, point_list, point_data, atol_cof=0.03):
        for point in point_list.copy():
            if np.allclose([point.x(), point.y()], [point_data.x(), point_data.y()], atol=atol_cof):
                point_list.remove(point)
                point_pair = pg.Point(point.x(), -point.y())
                if point_pair in point_list:
                    point_list.remove(point_pair)
                self.update_plot()
                return True
        self.update_plot()
        return False

    def clear_plots(self):
        self.data = [0, 0]
        self.data_modified = [0, 0]
        self.inputSignal.setData([0])
        self.filteredSignal.setData([0])
        self.current_index = 0



    # Update Zeros, Poles Plotting
    def update_plot(self):
        self.unitCircleGraph.clear()

        for point_type in ["Zeros", "Poles"]:
            self.plot_zeroes_poles(self.data_dict[point_type], point_type)
        self.unitCircleGraph.addItem(self.roi_unitCircle)

        self.update_response_plots()
        self.update_plot_allpass()

    def plot_zeroes_poles(self, data, point_type):
        #data is a list of tuples (coordinates of zeroes and poles)
        for point in data:
            small_circle = pg.ScatterPlotItem(pos=[(point.x(), point.y())], brush=self.data_brush[point_type], size=10,
                                              symbol=self.data_symbol[point_type])
            self.unitCircleGraph.addItem(small_circle)



    # Update the Magnitude and Phase Response
    def update_response_plots(self):
        # Combine zeros and poles
        z, p, z_allpass, p_allpass = self.get_all_pass_filter()

        # Calculate frequency response
        w, h = freqz(np.poly(z), np.poly(p))
        #w: frequency(omega) , h : transferFunction = z/p

        # Update class attributes
        self.frequencies, self.mag_response, self.phase_response = w, np.abs(h), self.fix_phase(h)

        # Plot magnitude response
        self.plot_response(self.magnitudeGraph, self.frequencies, self.mag_response, pen='b', label='Magnitude',
                           units='Linear', unit_bot="Radians")

        # Plot phase response
        self.plot_response(self.phaseGraph, self.frequencies, self.phase_response, pen='r', label='Phase',
                           units='Degrees', unit_bot="Radians", name="Normal Phase Response")

        w, h = freqz(np.poly(z_allpass), np.poly(p_allpass))
        self.frequencies, self.mag_response, self.phase_response = w, np.abs(h), self.fix_phase(h)
        self.phaseGraph.plot(x=self.frequencies, y=self.phase_response, pen='y', name="AllPass Phase Response")

    def plot_response(self, plot, x, y, pen, label, units, unit_bot, name=""):
        plot.clear()
        plot.plot(x, y, pen=pen, name=name)
        # plot.setLabel('left', label, units=units)
        # plot.setLabel('bottom', label, units=unit_bot)
        plot.setLabel('left',label , units=units)
        plot.setLabel('bottom',"Frequency", units=unit_bot)
        self.phaseGraph.addLegend()

    def fix_phase(self, h):
        phase_response_deg = np.rad2deg(np.angle(h))
        phase_response_constrained = np.where(phase_response_deg < 0, phase_response_deg + 360, phase_response_deg)
        phase_response_constrained = np.where(phase_response_constrained > 180, phase_response_constrained - 360,
                                              phase_response_constrained)

        return phase_response_constrained



    # Check All Pass filter and filter the phase
    def get_all_pass_filter(self):
        self.checked_coeffs = [0.0]  # List to hold the selected coefficient values

        for row in range(self.table_coeff.rowCount()):
            item = self.table_coeff.item(row, 0)
            if item.checkState() == Qt.CheckState.Checked:
                self.checked_coeffs.append(float(item.text()))

        if not self.allpass_en:
            self.checked_coeffs = [0.0]

        self.all_pass_zeros = self.data_dict["Zeros"].copy()
        self.all_pass_poles = self.data_dict["Poles"].copy()

        w, all_pass_phs = 0, 0
        self.allPassGraph.clear()

        for i in range(len(self.checked_coeffs)):
            a = self.checked_coeffs[i]

            if a == 1:
                a = 0.99999999
            a = complex(a, 0)

            # Check if denominator is not zero before performing division to avoid division by zero.
            if np.abs(a) > 0:
                a_conj = 1 / np.conj(a)

                w, h = freqz([-np.conj(a), 1.0], [1.0, -a])
                all_pass_phs = np.add(np.angle(h), all_pass_phs)
                self.allPassGraph.plot(w, np.angle(h), pen=self.colors[i % len(self.colors)], name=f'All pass{a.real}')
                self.allPassGraph.setLabel('left', 'All Pass Phase', units='degrees')

                # Add points to lists
                self.all_pass_poles.append(pg.Point(a.real, a.imag))
                self.all_pass_zeros.append(pg.Point(a_conj.real, a_conj.imag))

        self.unitCircleGraph.clear()
        self.plot_zeroes_poles(self.all_pass_zeros, "Zeros")
        self.plot_zeroes_poles(self.all_pass_poles, "Poles")
        self.unitCircleGraph.addItem(self.roi_unitCircle)

        if len(self.checked_coeffs) > 1:
            self.allPassGraph.plot(w, all_pass_phs, pen=self.colors[-1], name='All pass Total')
        self.allPassGraph.addLegend()

        # Combine zeros and poles
        z_allpass = np.array([complex(zero.x(), zero.y()) for zero in self.all_pass_zeros])
        p_allpass = np.array([complex(pole.x(), pole.y()) for pole in self.all_pass_poles])

        z = np.array([complex(zero.x(), zero.y()) for zero in self.data_dict["Zeros"]])
        p = np.array([complex(pole.x(), pole.y()) for pole in self.data_dict["Poles"]])

        return z, p, z_allpass, p_allpass

    def update_plot_allpass(self):
        self.update_response_plots()
        _, _, z_allpass, p_allpass = self.get_all_pass_filter()
        # Calculate frequency response
        w, h = freqz(np.poly(z_allpass), np.poly(p_allpass))

        self.phase_response = self.fix_phase(h)



    # Real Time Data Filtering
    def filter_data(self):
        _, _, z_allpass, p_allpass = self.get_all_pass_filter()
        numerator, denominator = zpk2tf(z_allpass, p_allpass, 1)
        self.data_modified = np.real(lfilter(numerator, denominator, self.data))


    # Draw Data Using Mouse Movement
    def on_mouse_move(self, pos):
        if self.mouse_enable:
            if not self.data_opened:
                self.data = []
                self.data_modified = []
                self.data_opened = True

            # Convert the click position to data coordinates
            pos = self.unitCircleGraph.getViewBox().mapSceneToView(pos)

            # Convert x-coordinate to a signal value (adjust the scaling factor as needed)
            signal_value = np.sqrt(pos.x() ** 2 + pos.y() ** 2)

            self.counter_max += 1

            # Append the signal value to self.data
            self.data.append(signal_value)

            # Update the filtered data
            self.filter_data()

            # Update inputSignal and filteredSignal data
            self.inputSignal.setData(self.data[:self.counter_max])
            self.filteredSignal.setData(self.data_modified[:self.counter_max])

            # Adjust x_min and x_max for plotting
            x_max = len(self.data)
            x_min = max(0, x_max - 200)

            # SetRange for real-time plots
            self.InputGraph.setRange(xRange=[x_min, x_max])
            self.filteredGraph.setRange(xRange=[x_min, x_max])

        self.prev_mouse_pos = pos

    def reset_viewport_range(self):
        for plot in [self.InputGraph, self.filteredGraph]:
            plot.setRange(xRange=[0, 1000])
        #to reset x-axis to start from zero

    def remove_all(self):
        self.data_dict["Zeros"].clear()
        self.data_dict["Poles"].clear()
        self.data_modified = self.data
        self.update_plot()

    def remove_points(self, point_type):
        self.data_dict[point_type].clear()
        self.update_plot()

    def set_added(self, point_type):
        self.added = point_type

    def toggle_pair_mode(self):
        self.pair_mode = not self.pair_mode

    def toggle_all_pass(self):
        self.allpass_en = not self.allpass_en
        self.update_plot_allpass()
        self.update_response_plots()

    def toggle_mouse_drawing(self):
        self.mouse_enable = not self.mouse_enable
        self.reset_viewport_range()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = MainApp()
    MainWindow.show()
    sys.exit(app.exec_())
