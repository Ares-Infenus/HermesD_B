import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout

# Crear la clase principal de la ventana
class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ejemplo PyQt6")  # Título de la ventana
        self.setGeometry(100, 100, 300, 200)  # Posición y tamaño de la ventana

        # Crear una etiqueta
        self.label = QLabel("¡Hola, PyQt6!", self)

        # Crear un botón
        self.button = QPushButton("Cambiar Texto", self)
        self.button.clicked.connect(self.change_text)  # Conectar el botón a la función

        # Crear un layout vertical y añadir los widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        # Establecer el layout en la ventana principal
        self.setLayout(layout)

    # Función para cambiar el texto de la etiqueta
    def change_text(self):
        self.label.setText("Texto cambiado al hacer clic en el botón.")

# Función principal para ejecutar la aplicación
def main():
    app = QApplication(sys.argv)  # Crear una instancia de la aplicación
    window = App()  # Crear una instancia de la ventana
    window.show()  # Mostrar la ventana
    sys.exit(app.exec())  # Ejecutar el bucle de eventos de la aplicación

if __name__ == "__main__":
    main()