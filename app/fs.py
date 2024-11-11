import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import os
import threading
import pandas as pd
import genetic as ga
import pso as pso

dataset, target, model, method, columnas = None, None, None, None, None


def reset_ui():
    message_label.config(text="")
    btn_procesar.config(state="normal")

def iniciar_procesamiento():
    reset_ui()
    btn_procesar.config(default="disabled")  # Deshabilita el botón mientras se procesa

    # Guardar el método de optimización seleccionado y el modelo
    global target, model, method
    method = metodo_selector.get()
    model = modelo_selector.get()
    target = target_selector.get()

    message_label.config(text=f"Dataset: {os.path.basename(dataset)}, Método: {method}, Modelo: {model}, Target: {target}. Procesando...")
    message_label.pack(pady=5)
    # Ejecuta `procesar_video` en un hilo separado
    thread = threading.Thread(target=feature_selecion)
    thread.start()

def feature_selecion():
    try:
        # Llamar a la función principal de genetic.py
        # Pasar el modelo seleccionado a int
        model_id = ["Decision Tree", "Random Forest", "SVM", "KNN", "LR"].index(model)
        if method == "PSO":
            features, accuracy = pso.main(dataset, target, model_id)
        else:
            features, accuracy = ga.main(dataset, target, model_id)
        message_label.config(text=f"Las mejores variables son: {features}\nPrecisión: {accuracy}", justify="center")
        ventana.update_idletasks()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def seleccionar_dataset():
    dataset_path = filedialog.askopenfilename(
        title="Seleccionar dataset",
        filetypes=[("Archivos datset", "*.csv")]
    )
    if dataset_path:
        message_label.config(text=f"Dataset seleccionado: {os.path.basename(dataset_path)}")
        global dataset, columnas
        dataset = dataset_path   
        columnas = leer_dataset(dataset_path)     
        return dataset_path
    return None

#Funcion que lee el dataset y devuelve una lista con las columnas
def leer_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    columnas = list(dataset.columns)
    # Actualizar el combobox de las columnas
    target_selector.config(values=columnas)
    target_selector.set(columnas[0])
    return columnas

if __name__ == "__main__":
    # Crear ventana con ttkbootstrap
    ventana = ttk.Window(themename="darkly")
    ventana.title("Feature Selection")
    ventana.geometry("800x700")

    # Nombre de la aplicación
    app_name = ttk.Label(ventana, text="Feature Selection", font=("Arial", 20))
    app_name.pack(pady=5)


    # Cargar el logo
    logo_image = Image.open("./app/logo.png")
    logo_image = logo_image.resize((150, 150))
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(ventana, image=logo_photo)
    logo_label.pack(pady=10)

    # Título
    titulo_label = ttk.Label(ventana, text="Sube tu dataset para descubrir las mejores variables", font=("Arial", 16))
    titulo_label.pack(pady=10)

    # Botón para seleccionar el video
    btn_ds = ttk.Button(ventana, text="Seleccionar dataset", bootstyle=PRIMARY, command=seleccionar_dataset, width=35)
    btn_ds.pack(pady=10)

    # Combobox para seleccionar PSO o GA
    metodo_label = ttk.Label(ventana, text="Selecciona el método de optimización", font=("Arial", 12))
    metodo_label.pack(pady=5)
    metodo_selector = ttk.Combobox(ventana, values=["PSO", "GA"], state="readonly", width=35)
    metodo_selector.set("PSO")  
    metodo_selector.pack(pady=5)

    #Combobox para seleccionar el modelo
    modelo_label = ttk.Label(ventana, text="Selecciona el modelo", font=("Arial", 12))
    modelo_label.pack(pady=5)
    modelo_selector = ttk.Combobox(ventana, values=["Decision Tree", "Random Forest", "SVM", "KNN", "LR"], state="readonly", width=35)
    modelo_selector.set("Decision Tree")  
    modelo_selector.pack(pady=5)

    # Combobox para seleccionar el target
    target_label = ttk.Label(ventana, text="Selecciona el la variable objetivo", font=("Arial", 12))
    target_label.pack(pady=5)
    target_selector = ttk.Combobox(ventana, values="", state="readonly", width=35)
    target_selector.set("Sube un dataset para ver las variables")
    target_selector.pack(pady=5)


    # Botón para iniciar el procesamiento
    btn_procesar = ttk.Button(ventana, text="Iniciar Feature Selection", bootstyle=SUCCESS, width=35)
    btn_procesar.pack(pady=10)
    btn_procesar.config(command=iniciar_procesamiento)

    # Label para mostrar mensajes de estado
    message_label = ttk.Label(ventana, text="", font=("Arial", 10))
    message_label.pack(pady=5)

    
    ventana.mainloop()