import sys
from PyQt5.QtWidgets import QApplication
from frontend.main_window import NarrativeNavigator

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply QSS stylesheet
    try:
        with open('style.qss', 'r') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("Warning: 'style.qss' not found. UI will not be styled.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading stylesheet: {e}", file=sys.stderr)

    window = NarrativeNavigator()
    window.show()
    sys.exit(app.exec_())