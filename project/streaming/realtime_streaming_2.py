import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import os
import time
from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt

import numpy as np
from streaming.prod_dca_2 import producer_real_time_1843

def plot_2d_heatmap(ax, data, theta, r, vmin=0, vmax=0.1):
    R, Theta = np.meshgrid(r, theta)
    ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlim(theta[0], theta[-1])
    ax.set_ylim(r[0], r[-1])
    ax.grid(False)

def consumer(q, index):
    app = MyApp(q)
    app.run()

class MyApp(ShowBase):
    def __init__(self, queue):
        ShowBase.__init__(self)
        self.q = queue
        self.latest_msg = None
        self.phi = np.linspace(0, np.pi, 180)
        self.r_idxs = np.arange(0, 120)
        self.bev_map = np.zeros((len(self.phi), len(self.r_idxs)))

        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self._configure_ax()
        plt.show(block=False)

        self.taskMgr.add(self.updateTask, "updateTask")

    def _configure_ax(self):
        self.ax.set_theta_zero_location('E')
        self.ax.set_theta_direction(1)
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(180)
        self.ax.set_title("Bird Eye View (Top View)")

    def updateTask(self, task):
        try:
            while not self.q.empty():
                msg = self.q.get_nowait()
                if msg[0] == "bev":
                    self.latest_msg = msg[1]
        except:
            pass

        if self.latest_msg:
            self.phi, self.r_idxs, self.bev_map = self.latest_msg
            self.ax.clear()
            self._configure_ax()
            plot_2d_heatmap(self.ax, self.bev_map, self.phi, self.r_idxs, vmin=0, vmax=0.1)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        return Task.cont

def main(exp_num, lua_file):
    q_main = Queue(maxsize=1)  # ❗️ Only keep latest

    producers = [Process(target=producer_real_time_1843, args=(q_main, 0, lua_file), daemon=True)]
    consumers = [Process(target=consumer, args=(q_main, 0), daemon=True)]

    for p in producers: p.start()
    for c in consumers: c.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in producers: p.terminate()
        for c in consumers: c.terminate()
        for p in producers: p.join()
        for c in consumers: c.join()
        print("✅ Shutdown complete.")

