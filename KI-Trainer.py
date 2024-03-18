import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import shutil
from flask import Flask, request, jsonify

# Model Trainer V 0.0.1 by Manfred Aabye

# Abhängigkeiten
##pip install torch
##pip install tensorflow
##pip install datasets
##pip install transformers bitsandbytes
##
##pip install scikit-learn
##pip install matplotlib
##pip install flask

class ModelTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer")
        self.loaded_model = None

        # Menü Datei erstellen
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Modell laden", command=self.load_model)
        file_menu.add_command(label="Modell speichern", command=self.save_model)
        file_menu.add_separator()
        file_menu.add_command(label="Text/PDF laden", command=self.load_text_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Beenden", command=self.exit_program)
        menubar.add_cascade(label="Datei", menu=file_menu)

        # Weitere Optionen
        additional_menu = tk.Menu(menubar, tearoff=0)
        additional_menu.add_command(label="Hyperparameter-Tuning", command=self.hyperparameter_tuning)
        additional_menu.add_command(label="Auswertung", command=self.evaluate_model)
        additional_menu.add_command(label="Visualisierung", command=self.visualize_model)
        additional_menu.add_command(label="Transfer Learning", command=self.transfer_learning)
        additional_menu.add_command(label="Modellinterpretation", command=self.model_interpretation)
        additional_menu.add_command(label="Exportieren", command=self.export_model)
        additional_menu.add_command(label="Versionierung", command=self.model_versioning)
        additional_menu.add_command(label="Online Deployment", command=self.online_deployment)
        additional_menu.add_command(label="Modellvalidierung", command=self.validate_model)
        additional_menu.add_command(label="Datenexploration", command=self.explore_data)
        menubar.add_cascade(label="Weitere Optionen", menu=additional_menu)

        self.root.config(menu=menubar)

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Modell-Dateien", "*.pt")])
        if file_path:
            try:
                self.loaded_model = torch.load(file_path)
                messagebox.showinfo("Erfolg", "Modell erfolgreich geladen!")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden des Modells: {e}")

    def save_model(self):
        if self.loaded_model:
            file_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("Modell-Dateien", "*.pt")])
            if file_path:
                try:
                    torch.save(self.loaded_model, file_path)
                    messagebox.showinfo("Erfolg", "Modell erfolgreich gespeichert!")
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Speichern des Modells: {e}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def load_text_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text/PDF-Dateien", "*.txt;*.pdf")])
        if file_path:
            try:
                dataset = load_dataset("text", data_files=file_path)
                text_data = dataset["train"]["text"]
                messagebox.showinfo("Erfolg", "Text/PDF-Dateien erfolgreich geladen und konvertiert!")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden und Konvertieren der Text/PDF-Dateien: {e}")

    def hyperparameter_tuning(self):
        # Beispiel: Meldung mit ausgewählten Hyperparametern anzeigen
        hyperparameters = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'num_epochs': 10
        }
        messagebox.showinfo("Hyperparameter-Tuning", f"Ausgewählte Hyperparameter: {hyperparameters}")

    def evaluate_model(self):
        if self.loaded_model:
            # Beispiel: Auswertung des Modells mit Dummy-Daten
            test_data = torch.randn(100, 10)
            true_labels = torch.randint(0, 2, (100,))
            predicted_labels = self.loaded_model(test_data)
            accuracy = accuracy_score(true_labels, predicted_labels.argmax(dim=1))
            messagebox.showinfo("Auswertung", f"Genauigkeit: {accuracy:.2f}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def visualize_model(self):
        if self.loaded_model:
            # Beispiel: Visualisierung der Gewichte der ersten Schicht
            if isinstance(self.loaded_model, nn.Module):
                first_layer_weights = self.loaded_model[0].weight.data.numpy()
                plt.imshow(first_layer_weights, cmap='viridis', aspect='auto')
                plt.title('Visualisierung der Gewichte der ersten Schicht')
                plt.colorbar()
                plt.show()
            else:
                messagebox.showerror("Fehler", "Das Modell unterstützt keine Visualisierung.")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def transfer_learning(self):
        if self.loaded_model:
            # Lade ein vortrainiertes Modell (hier als Beispiel ResNet)
            pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

            # Einfrieren der vorherigen Schichten, um Gewichte nicht zu verlieren
            for param in pretrained_model.parameters():
                param.requires_grad = False

            # Ersetzen des Klassifikations-Layers am Ende
            num_ftrs = pretrained_model.fc.in_features
            pretrained_model.fc = nn.Linear(num_ftrs, num_classes)  # Ändere num_classes entsprechend deinem Datensatz

            # Optionales Fine-Tuning
            for param in pretrained_model.fc.parameters():
                param.requires_grad = True

            self.loaded_model = pretrained_model
            messagebox.showinfo("Info", "Transfer Learning erfolgreich angewendet!")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def model_interpretation(self):
        if self.loaded_model:
            if isinstance(self.loaded_model, nn.Module):
                # Beispiel: Visualisierung der Gewichte der ersten Schicht
                if hasattr(self.loaded_model, 'weight'):
                    weights = self.loaded_model.weight.data.numpy()
                    plt.bar(range(len(weights)), weights)
                    plt.title('Gewichte der ersten Schicht')
                    plt.xlabel('Neuron')
                    plt.ylabel('Gewicht')
                    plt.show()
                else:
                    messagebox.showerror("Fehler", "Das Modell hat keine interpretierbaren Gewichte.")
            else:
                messagebox.showerror("Fehler", "Das geladene Objekt ist kein neuronales Netz.")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def export_model(self):
        if self.loaded_model:
            file_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("Modell-Dateien", "*.pt")])
            if file_path:
                try:
                    torch.save(self.loaded_model, file_path)
                    messagebox.showinfo("Erfolg", "Modell erfolgreich exportiert!")
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Exportieren des Modells: {e}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def model_versioning(self):
        if self.loaded_model:
            try:
                # Speichern des aktuellen Modells mit der aktuellen Version
                model_path = f"model_v{self.model_version}.pt"
                torch.save(self.loaded_model, model_path)
                
                messagebox.showinfo("Info", f"Modell erfolgreich versioniert als Version {self.model_version}!")
                
                # Inkrementiere die Version für das nächste Mal
                self.model_version += 1
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Versionieren des Modells: {e}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def online_deployment(self):
        if self.loaded_model:
            # Starte einen einfachen Flask-Server
            app = Flask(__name__)

            @app.route('/predict', methods=['POST'])
            def predict():
                data = request.get_json(force=True)
                input_data = torch.tensor(data['input'])
                output = self.loaded_model(input_data).tolist()
                return jsonify(output)

            app.run(host='0.0.0.0', port=5000)  # Starte den Server

        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def validate_model(self):
        if self.loaded_model:
            try:
                # Beispiel: Laden der Testdaten
                test_data = torch.randn(100, 10)
                true_labels = torch.randint(0, 2, (100,))

                # Anwendung des Modells auf die Testdaten
                predicted_labels = self.loaded_model(test_data)

                # Berechnung der Genauigkeit
                accuracy = accuracy_score(true_labels, predicted_labels.argmax(dim=1))
                messagebox.showinfo("Auswertung", f"Genauigkeit: {accuracy:.2f}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Validierung des Modells: {e}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")

    def explore_data(self):
        if self.loaded_model:
            try:
                # Beispiel: Laden des Datensatzes
                dataset = load_dataset("text")

                # Anzeigen einiger Beispieldaten
                print("Beispielhafte Eingabe:")
                print(dataset["train"]["text"][:5])

                # Visualisierung der Daten (hier als Beispiel für Textdaten)
                text_lengths = [len(text.split()) for text in dataset["train"]["text"]]
                plt.hist(text_lengths, bins=30)
                plt.title("Verteilung der Textlängen")
                plt.xlabel("Anzahl der Wörter")
                plt.ylabel("Häufigkeit")
                plt.show()
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler bei der Exploration der Daten: {e}")
        else:
            messagebox.showerror("Fehler", "Es wurde kein Modell geladen.")


    def exit_program(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")  # 800x600 Pixel
    app = ModelTrainerApp(root)
    root.mainloop()
