from tkinter import ttk, filedialog, messagebox, font
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tensorflow import keras
import numpy as np

import os
import sys

def resource_path(path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, path)

def main_screen():
    model = keras.models.load_model(resource_path('modelo_cnn.keras'))

    window = ThemedTk(theme="adapta")
    window.title('GPT')
    window.geometry("270x280")
    window.resizable(False, False)

    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

    img_size = (224, 224)
    current_image = None

    def change_image(img_size, img_label):
        nonlocal current_image
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Escolha um arquivo",
                                              filetypes=[("Imagem", "*.png *.jpg *.jpeg")])
        if not filename:
            return

        image = keras.utils.load_img(filename, target_size=img_size)
        current_image = image
        photo_image = ImageTk.PhotoImage(image)
        img_label.configure(image=photo_image)
        img_label.photo = photo_image

    def check_image(model, img):
        if img is None:
            messagebox.showerror("Erro", "Selecione uma imagem")
            return

        raw_image = keras.utils.img_to_array(img)
        raw_image /= 255
        input_arr = np.array([raw_image])
        prediction = model.predict(input_arr)[0][0]

        if prediction >= 0.95:
            messagebox.showinfo("Resultado", "Tumor detectado")
        else:
            messagebox.showinfo("Resultado", "Ausência de tumor.")

    tk_image = ImageTk.PhotoImage(
        Image.new(mode="RGB", size=img_size)
            if current_image is None else current_image
    )

    img_label = ttk.Label(image=tk_image)
    img_label.grid(column=0, row=0, padx=(20,20), pady=(5,5))

    button_explore = ttk.Button(window,
                                text="Selecionar arquivo",
                                command=lambda: change_image(img_size, img_label))
    button_explore.grid(column = 0, row = 1, rowspan = 1, padx=(20,0), sticky='W')

    button_check = ttk.Button(window,
                              text="Checar",
                              command=lambda: check_image(model, current_image))
    button_check.grid(column = 0, row = 1, rowspan = 1, padx=(0,20), sticky='E')


    window.mainloop()


def launch_app(home):
    home.destroy() 
    main_screen() 


def home_screen():
    home = ThemedTk(theme="adapta")
    home.title("GPT")
    home.geometry("270x280")
    home.resizable(False, False)

    home.update_idletasks()
    width = home.winfo_width()
    height = home.winfo_height()
    x = (home.winfo_screenwidth() // 2) - (width // 2)
    y = (home.winfo_screenheight() // 2) - (height // 2)
    home.geometry(f'{width}x{height}+{x}+{y}')

    title_source = font.Font(family="Helvetica", size=36, weight="bold")
    subtitle_source = font.Font(family="Helvetica", size=11)

    frame = ttk.Frame(home, padding="20")
    frame.pack(expand=True, fill="both")

    title_label = ttk.Label(frame, text="GPT", font=title_source)
    title_label.pack(pady=(50, 0))

    subtitle_slabel = ttk.Label(frame, text="Grading Prediction of Tumors", font=subtitle_source)
    subtitle_slabel.pack(pady=(0, 20))

    start_button = ttk.Button(frame, text="Iniciar", command=lambda: launch_app(home))
    start_button.pack(pady=10, ipadx=20) 

    home.mainloop()


if __name__ == "__main__":
    home_screen()
