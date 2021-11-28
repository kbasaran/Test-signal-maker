import sys

from PySide2 import QtWidgets as qtw
from PySide2 import QtCore as qtc

qapp = qtw.QApplication.instance()
if not qapp:
    qapp = qtw.QApplication()

mw = qtw.QWidget()
layout = qtw.QVBoxLayout()
label = qtw.QLabel()
layout.addWidget(label)
mw.setLayout(layout)

settings = qtc.QSettings('kbasaran', 'Test signal maker')

label_text = "Data in settings:\n\n"
for key in settings.allKeys():
    label_text += f"{key}: {settings.value(key)}\n"

# Clear settings
settings.clear()
label_text += "\nAll stored settings cleared."

label.setText(label_text)
mw.show()
sys.exit(qapp.exec_())
