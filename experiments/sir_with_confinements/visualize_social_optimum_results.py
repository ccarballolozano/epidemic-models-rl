import argparse
import glob
import tkinter as tk
from tkinter import ttk


import numpy as np
from PIL import Image, ImageTk


class ImageApp:
    def __init__(self, master):
        self.master = master

        self.N = tk.IntVar(value=N_[0])
        self.encounter_prob_N = tk.DoubleVar(
            value=encounter_prob_N_[0]
        )  # Default value for the first part of the number
        self.recovery_prob_N = tk.DoubleVar(
            value=recovery_prob_N_[0]
        )  # Default value for the second part of the number
        self.cost_infection = tk.DoubleVar(value=cost_infection_[0])
        self.cost_lockdown = tk.DoubleVar(value=cost_lockdown_[0])

        # add spinbox for N
        self.label_N = ttk.Label(self.master, text="N")
        self.label_N.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_N = ttk.Spinbox(
            self.master, values=N_, command=self.load_image, textvariable=self.N
        )
        self.spinbox_N.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_encounter_prob = ttk.Label(self.master, text="$ N \gamma $")
        self.label_encounter_prob.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_encounter_prob = ttk.Spinbox(
            self.master,
            values=encounter_prob_N_,
            command=self.load_image,
            textvariable=self.encounter_prob_N,
        )
        self.spinbox_encounter_prob.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_recovery_prob = ttk.Label(self.master, text="$ N \beta $")
        self.label_recovery_prob.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_recovery_prob = ttk.Spinbox(
            self.master,
            values=recovery_prob_N_,
            command=self.load_image,
            textvariable=self.recovery_prob_N,
        )
        self.spinbox_recovery_prob.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_cost_infection = ttk.Label(self.master, text="Cost of infection")
        self.label_cost_infection.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_cost_infection = ttk.Spinbox(
            self.master,
            values=cost_infection_,
            command=self.load_image,
            textvariable=self.cost_infection,
        )
        self.spinbox_cost_infection.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_cost_lockdown = ttk.Label(self.master, text="Cost of lockdown")
        self.label_cost_lockdown.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_cost_lockdown = ttk.Spinbox(
            self.master,
            values=cost_lockdown_,
            command=self.load_image,
            textvariable=self.cost_lockdown,
        )
        self.spinbox_cost_lockdown.pack(side=tk.TOP, padx=(0, 10), pady=5)

        # Canvas to display the image
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack()

        # Load the initial image
        self.load_image()

        # Bind the sliders to update the image when their values change
        # self.slider.bind("<ButtonRelease-1>", self.load_image)
        # self.slider_2.bind("<ButtonRelease-1>", self.load_image)

    def load_image(self, event=None):
        try:
            # Construct the image path based on the selected image numbers
            image_path = f"{images_dir}/plt_{self.N.get()}_{self.encounter_prob_N.get()}_{self.recovery_prob_N.get()}_{self.cost_infection.get()}_{self.cost_lockdown.get()}.png"
            image = Image.open(image_path)
            photo = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference to prevent garbage collection
        except Exception as e:
            print("Error loading image:", e)


def main(args):
    global images_dir
    global N_, encounter_prob_N_, recovery_prob_N_, cost_infection_, cost_lockdown_
    images_dir = args.images_dir

    image_names = list(glob.glob(f"{images_dir}/plt_*.png"))
    params = [
        image_name.split("/")[-1].split(".png")[0].split("_")[1:]
        for image_name in image_names
    ]
    N_ = sorted(list(set([int(param[0]) for param in params])))
    encounter_prob_N_ = sorted(list(set([float(param[1]) for param in params])))
    recovery_prob_N_ = sorted(list(set([float(param[2]) for param in params])))
    cost_infection_ = sorted(list(set([float(param[3]) for param in params])))
    cost_lockdown_ = sorted(list(set([float(param[4]) for param in params])))

    root = tk.Tk()
    root.title("Image Viewer")

    app = ImageApp(root)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="outputs/social_optimum/2024-04-29 02:18:22.639834",
    )
    args = parser.parse_args()
    main(args)
