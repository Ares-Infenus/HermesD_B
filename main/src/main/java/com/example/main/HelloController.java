package com.example.main;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.application.Platform; // Para cerrar la aplicación

public class HelloController {
    @FXML
    private Label welcomeText;

    @FXML
    protected void onHelloButtonClick() {
        welcomeText.setText("Welcome to JavaFX Application!");
    }

    // Método para salir de la aplicación
    @FXML
    protected void handleExitAction() {
        Platform.exit();  // Cierra la aplicación
    }
}