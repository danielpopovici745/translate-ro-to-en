from gui import audioApp
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv) # Initialize GUI application
    window = audioApp()
    # Instantiate the audioApp class
    window.show()
    # Show the main window
    sys.exit(app.exec_())
    # When the app is closed, exit the program