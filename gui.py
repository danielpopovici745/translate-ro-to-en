from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import pyaudio
import threading
from liveTranscription import main  # Import the transcription logic

class transcriptionThread(QThread):
    """Thread to run the transcription logic."""
    transcription_finished = pyqtSignal()

    def __init__(self, input_device, output_device):
        super().__init__()
        self.input_device = input_device
        self.output_device = output_device
        self.stop_event = threading.Event()  # Create a stop event

    def run(self):
        try:
            main(stop_event=self.stop_event, input_device=self.input_device, output_device=self.output_device)
        except Exception as e:
            print(f"Error in transcription thread: {e}")
        finally:
            self.transcription_finished.emit()

    def stop(self):
        self.stop_event.set()  # Signal the stop event

# This is the main GUI class
# It creates a window with input and output device selectors and start/stop buttons.
class audioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("translateRo2En")
        self.setGeometry(100, 100, 400, 200) # Set the initial size of the window
        self.setFixedSize(400, 200)  # Prevent resizing by setting a fixed size

        # Main layout
        layout = QVBoxLayout()

        # Input device selector
        self.input_label = QLabel("Select Input Device:")
        layout.addWidget(self.input_label) # Add label for input device selector
        self.input_selector = QComboBox() # Create a combo box for input device selection
        layout.addWidget(self.input_selector) # Add the combo box to the layout

        # Output device selector
        self.output_label = QLabel("Select Output Device:")
        layout.addWidget(self.output_label)
        self.output_selector = QComboBox()
        layout.addWidget(self.output_selector)

        # Start/Stop buttons
        self.start_button = QPushButton("Start Transcription") # Create a button to start transcription
        self.start_button.clicked.connect(self.start_transcription) # Connect the button to the start_transcription method
        layout.addWidget(self.start_button) # Add the start button to the layout

        self.stop_button = QPushButton("Stop Transcription") # Create a button to stop transcription
        self.stop_button.clicked.connect(self.stop_transcription) # Connect the button to the stop_transcription method
        self.stop_button.setEnabled(False) # Initially disable the stop button
        layout.addWidget(self.stop_button) # Add the stop button to the layout

        central_widget = QWidget() # Create a central widget for the main window
        central_widget.setLayout(layout) # Set the layout for the central widget
        self.setCentralWidget(central_widget) # Set the central widget for the main window
        self.input_label.setStyleSheet("""
            QLabel {
                color: #ffffff;  /* White text for the label */
                font-size: 14px;  /* Font size for the label */
            }""")
        self.output_label.setStyleSheet(""" QLabel {
            color: #ffffff;  /* White text for the label */
            font-size: 14px;  /* Font size for the label */
        }""")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #303030;  /* Dark gray background for the main window */
            }
        """)

        self.input_selector.setStyleSheet("""
            QComboBox {
                background-color: #404040;  /* Dark gray background for the main combo box */
                color: #ffffff;  /* White text for the main combo box */
                border: 1px solid #606060;  /* Border color */
            }
            QComboBox QAbstractItemView {
                background-color: #404040;  /* Dark gray background for the dropdown list */
                color: #ffffff;  /* White text for the dropdown list */
                selection-background-color: #606060;  /* Slightly lighter gray for the selected item */
                selection-color: #ffffff;  /* White text for the selected item */
            }
        """)

        self.output_selector.setStyleSheet("""
            QComboBox {
                background-color: #404040;  /* Dark gray background for the main combo box */
                color: #ffffff;  /* White text for the main combo box */
                border: 1px solid #606060;  /* Border color */
            }
            QComboBox QAbstractItemView {
                background-color: #404040;  /* Dark gray background for the dropdown list */
                color: #ffffff;  /* White text for the dropdown list */
                selection-background-color: #606060;  /* Slightly lighter gray for the selected item */
                selection-color: #ffffff;  /* White text for the selected item */
            }
        """)

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #91f272;  /* green background for the start button */
        }""")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;  /* dark background for the stop button */
        }""")

        # Populate device lists
        self.populate_devices()

        # Transcription thread
        self.transcription_thread = None

    def populate_devices(self):
        p = pyaudio.PyAudio()
        added_devices = set()  # Track already-added devices to avoid duplicates

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            full_name = device_info["name"]

            if full_name not in added_devices:  # Add only if not already added
                self.input_selector.addItem(full_name, full_name)
                self.output_selector.addItem(full_name, full_name)
                added_devices.add(full_name)

        p.terminate()

    def start_transcription(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f11919;  /* red background for the stop button */
        }""")
        self.start_button.setStyleSheet(""" QPushButton {
            background-color: #404040;  /* green background for the start button */
        }""")

        # Get the full selected input and output device names
        selected_input_device = self.input_selector.currentData()
        selected_output_device = self.output_selector.currentData()

        # Start the transcription thread and pass the selected devices
        self.transcription_thread = transcriptionThread(selected_input_device, selected_output_device)
        self.transcription_thread.transcription_finished.connect(self.cleanup_thread)
        self.transcription_thread.start()

    def stop_transcription(self):
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(""" QPushButton {
            background-color: #404040;  /* dark background for the start button */
        }""")
        # Stop the transcription thread
        if self.transcription_thread:
            self.transcription_thread.stop()  # Signal the thread to stop
            self.transcription_thread.transcription_finished.connect(self.cleanup_thread)
            

    def cleanup_thread(self):
        """Clean up the transcription thread after it finishes."""
        self.transcription_thread = None
        self.start_button.setEnabled(True)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #91f272;  /* green background for the stop button */
            }""")