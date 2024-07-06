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
        self.encounter_rate = tk.DoubleVar(
            value=encounter_rate_[0]
        )  # Default value for the first part of the number
        self.recovery_rate = tk.DoubleVar(
            value=recovery_rate_[0]
        )  # Default value for the second part of the number
        self.susceptible_rate = tk.DoubleVar(value=susceptible_rate_[0])
        self.vaccination_rate = tk.DoubleVar(value=vaccination_rate_[0])
        self.cost_infection = tk.DoubleVar(value=cost_infection_[0])
        self.cost_lockdown = tk.IntVar(value=cost_lockdown_[0])

        # add spinbox for N
        self.label_N = ttk.Label(self.master, text="population (N + 1)")
        self.label_N.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_N = ttk.Spinbox(
            self.master, values=N_, command=self.load_image, textvariable=self.N
        )
        self.spinbox_N.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_encounter_rate = ttk.Label(self.master, text="encounter (gamma)")
        self.label_encounter_rate.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_encounter_rate = ttk.Spinbox(
            self.master,
            values=encounter_rate_,
            command=self.load_image,
            textvariable=self.encounter_rate,
        )
        self.spinbox_encounter_rate.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_recovery_rate = ttk.Label(self.master, text="recovery (rho)")
        self.label_recovery_rate.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_recovery_rate = ttk.Spinbox(
            self.master,
            values=recovery_rate_,
            command=self.load_image,
            textvariable=self.recovery_rate,
        )
        self.spinbox_recovery_rate.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_susceptible_rate = ttk.Label(
            self.master, text="re-susceptible (beta)"
        )
        self.label_susceptible_rate.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_susceptible_rate = ttk.Spinbox(
            self.master,
            values=susceptible_rate_,
            command=self.load_image,
            textvariable=self.susceptible_rate,
        )
        self.spinbox_susceptible_rate.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_vaccination_rate = ttk.Label(self.master, text="vaccination (alpha)")
        self.label_vaccination_rate.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_vaccination_rate = ttk.Spinbox(
            self.master,
            values=vaccination_rate_,
            command=self.load_image,
            textvariable=self.vaccination_rate,
        )
        self.spinbox_vaccination_rate.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_cost_infection = ttk.Label(
            self.master, text="Cost of infection (c_I)"
        )
        self.label_cost_infection.pack(side=tk.TOP, padx=(10, 5), pady=5)
        self.spinbox_cost_infection = ttk.Spinbox(
            self.master,
            values=cost_infection_,
            command=self.load_image,
            textvariable=self.cost_infection,
        )
        self.spinbox_cost_infection.pack(side=tk.TOP, padx=(0, 10), pady=5)

        self.label_cost_lockdown = ttk.Label(self.master, text="Cost of lockdown (c_L)")
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
            image_path = f"{images_dir}/plt_{self.N.get()}_{float_to_int(self.encounter_rate.get())}_{float_to_int(self.recovery_rate.get())}_{float_to_int(self.susceptible_rate.get())}_{float_to_int(self.vaccination_rate.get())}_{float_to_int(self.cost_infection.get())}_{float_to_int(self.cost_lockdown.get())}.png"
            image = Image.open(image_path)
            photo = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference to prevent garbage collection
        except Exception as e:
            print("Error loading image:", e)


def float_to_int(num):
    if int(num) == num:
        return int(num)
    else:
        return num


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def main(args):
    global images_dir
    global N_, encounter_rate_, recovery_rate_, susceptible_rate_, vaccination_rate_, cost_infection_, cost_lockdown_
    images_dir = args.images_dir

    image_names = list(glob.glob(f"{images_dir}/plt_*.png"))
    params = [
        image_name.split("/")[-1].split(".png")[0].split("_")[1:]
        for image_name in image_names
    ]
    N_ = sorted(list(set([str_to_num(param[0]) for param in params])))
    encounter_rate_ = sorted(list(set([str_to_num(param[1]) for param in params])))
    recovery_rate_ = sorted(list(set([str_to_num(param[2]) for param in params])))
    susceptible_rate_ = sorted(list(set([str_to_num(param[3]) for param in params])))  #
    vaccination_rate_ = sorted(list(set([str_to_num(param[4]) for param in params])))  #
    cost_infection_ = sorted(list(set([str_to_num(param[5]) for param in params])))
    cost_lockdown_ = sorted(list(set([str_to_num(param[6]) for param in params])))

    root = tk.Tk()
    root.title("Image Viewer")

    app = ImageApp(root)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="outputs/social_optimum/20240626",
    )
    args = parser.parse_args()
    main(args)
