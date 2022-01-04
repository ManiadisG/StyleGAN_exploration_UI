import argparse
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import sys

sys.path.append(os.getcwd())

from model import StyleGan

plt.ioff()

class StyleGANVisualizer(tk.Tk):
    def __init__(self, weights_path=None, sliders_no=20, gpu=0):
        tk.Tk.__init__(self)
        self.gpu=gpu
        self.sliders_no=sliders_no
        self.configure(bg='white')
        self.GAN = StyleGan(weights_path, gpu)
        self.GAN.reset_latent()
        print("\nStyleGAN loaded.")
        self.slider_job = None
        self.slider_to_feature_match = {}
        self.feature_index_change = 0
        self.active_slider = 0
        self.samples_plot_frame=None

        self.make_latent_space_frame()
        self.make_sample_frame()

    def make_latent_space_frame(self):
        self.latent_space_frame = tk.Frame(self, bg='white', relief=tk.RAISED, borderwidth=2)
        self.latent_space_frame_widgets = {}
        self.latent_space_frame_widgets['reset_latent_button'] = tk.Button(self.latent_space_frame,
                                                                           text="New Latent Vector",
                                                                           command=self.reset_latent_vector)
        self.latent_space_frame_widgets['reset_latent_button'].pack()
        for i in range(self.sliders_no):
            self.slider_to_feature_match[i] = i
            self.latent_space_frame_widgets['slider_' + str(i) + '_frame'] = tk.Frame(self.latent_space_frame,
                                                                                      bg='white', relief=tk.RAISED,
                                                                                      borderwidth=2)
            self.latent_space_frame_widgets.update(
                self.make_latent_vector_slider(self.latent_space_frame_widgets['slider_' + str(i) + '_frame'], i, i))
            self.latent_space_frame_widgets['slider_' + str(i) + '_frame'].pack()
        self.latent_space_frame.pack(side=tk.LEFT)

    def make_latent_vector_slider(self, parent_frame, slider_no, feature_index):
        label_1 = tk.Label(parent_frame, text="Feature No:", bg='white')
        entry = tk.Entry(parent_frame, width=32)
        entry.insert(tk.END, str(feature_index))
        label_2 = tk.Label(parent_frame, text="Feature Value:", bg='white')
        slider = tk.Scale(parent_frame, from_=-4, to=4, command=self.get_slider_function(feature_index), bg='white',
                          orient=tk.HORIZONTAL, length=200, resolution=0.01)
        slider.set(self.get_latent_feature(feature_index))
        button_choose_feature = tk.Button(parent_frame, text="Choose Feature",
                                          command=self.get_button_cf_function(entry, slider, feature_index))
        button_add = tk.Button(parent_frame, text="+",
                               command=self.get_button_add_function(slider))
        button_sub = tk.Button(parent_frame, text="-",
                               command=self.get_button_sub_function(slider))
        label_1.pack(side=tk.LEFT)
        entry.pack(side=tk.LEFT)
        button_choose_feature.pack(side=tk.LEFT)
        label_2.pack(side=tk.LEFT)
        slider.pack(side=tk.LEFT)
        button_add.pack(side=tk.LEFT)
        button_sub.pack(side=tk.LEFT)
        return {'slider_' + str(slider_no) + '_label_1': label_1, 'slider_' + str(slider_no) + '_entry': entry,
                'slider_' + str(slider_no) + '_label_2': label_2, 'slider_' + str(slider_no) + '_slider': slider,
                'slider_' + str(slider_no) + '_button': button_choose_feature,
                'slider_' + str(slider_no) + '_add_button': button_add,
                'slider_' + str(slider_no) + '_sub_button': button_sub}

    def get_button_cf_function(self, entry, slider, feature_index):
        def button_cf_function():
            v = int(round(float(entry.get())))
            if v>=512:
                entry.delete(0, tk.END)
                entry.insert(tk.END, "Error: Must be between 0 and 511")
            else:
                self.slider_to_feature_match[feature_index] = v
                slider.set(self.get_latent_feature(v))

        return button_cf_function

    def get_button_add_function(self, slider):
        def button_add_function():
            v = slider.get()
            slider.set(v + 0.01)
        return button_add_function

    def get_button_sub_function(self, slider):
        def button_sub_function():
            v = slider.get()
            slider.set(v - 0.01)
        return button_sub_function

    def get_slider_function(self, feature_index):
        def slider_function(event):
            if self.slider_job is not None:
                self.after_cancel(self.slider_job)
            self.feature_index_change = self.slider_to_feature_match[feature_index]
            self.active_slider = feature_index
            self.slider_job = self.after(200, self.change_latent_vector_slider)

        return slider_function

    def change_latent_vector_slider(self):
        self.slider_job = None
        self.GAN.latent[0, self.feature_index_change] = self.latent_space_frame_widgets[
            'slider_' + str(self.active_slider) + '_slider'].get()
        self.make_sample_frame()

    def make_sample_frame(self):
        fig = self.get_sample_plots()
        if self.samples_plot_frame is not None:
            self.samples_plot_frame.destroy()
        self.samples_plot_frame = tk.Frame(self, bg='white', relief=tk.RAISED, borderwidth=2)
        self.samples_plot_widgets = {}
        self.samples_plot_widgets['samples_canvas'] = FigureCanvasTkAgg(fig, master=self.samples_plot_frame)
        self.samples_plot_widgets['samples_canvas'].get_tk_widget().pack()
        self.samples_plot_widgets['samples_toolbar'] = NavigationToolbar2Tk(self.samples_plot_widgets['samples_canvas'], self.samples_plot_frame)
        self.samples_plot_widgets['samples_toolbar'].pack()
        self.samples_plot_frame.pack(side=tk.LEFT)

    def reset_latent_vector(self):
        self.GAN.reset_latent()
        for i in range(self.sliders_no):
            f = self.slider_to_feature_match[i]
            self.latent_space_frame_widgets['slider_' + str(i) + '_slider'].set(round(self.GAN.latent[0, f].item(), 2))

    def get_latent_feature(self, feature_index):
        return round(self.GAN.latent[0, feature_index].item(), 2)

    def get_sample_plots(self):
        img = self.GAN.fix_latent_sample()
        fig = plt.Figure(figsize=(9, 9), constrained_layout=True)
        gridspec = fig.add_gridspec(1)
        subfig=fig.add_subfigure(gridspec[0, :])
        subplot = subfig.add_subplot(1,1,1)
        subplot.imshow(img[0].cpu().numpy())
        return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--weights_path", default=None, type=str)
    parser.add_argument("--sliders_no", default=15, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    args = parser.parse_known_args()[0]

    app = StyleGANVisualizer(args.weights_path, args.sliders_no, args.gpu)
    app.mainloop()
    plt.close()
