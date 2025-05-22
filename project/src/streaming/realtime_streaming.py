import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from .prod_dca import producer_real_time_1843
from visualization.visualization import configure_ax_bf, configure_ax_db, configure_ax_gtrack, plot_2d_heatmap


def consumer(q, cfg_radar):
    app = MyApp(q, cfg_radar)
    app.run()

class MyApp(ShowBase):
    def __init__(self, queue, cfg_radar):
        ShowBase.__init__(self)
        self.q = queue
        self.latest_msg = None
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]

        #plt.ion() # Plus lent ??

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        configure_ax_bf(self.ax)

        self.fig_3 = plt.figure(figsize=(8, 6), constrained_layout=True)
        self.ax_3 = self.fig_3.add_subplot(111)

        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_text = self.ax.text(0.02, 1.02, "", transform=self.ax.transAxes, fontsize=10, color='blue')

        self.taskMgr.add(self.updateTask, "updateTask")


    def updateTask(self, task):
        new_msg = False

        try:
            while not self.q.empty():
                msg = self.q.get_nowait()
                if msg[0] == "bev":
                    self.latest_msg = msg[1]
                    new_msg = True

        except:
            pass

        if self.latest_msg and new_msg:
            # Unpack the latest message
            bf, gtrack = self.latest_msg

            # Update the beamforming plot
            self.ax.clear()
            configure_ax_bf(self.ax)
            plot_2d_heatmap(self.ax, bf, self.phi, self.r_idxs, vmin=0, vmax=0.1)

            # Update the gtrack plot
            self.ax_3.clear()
            tracks = gtrack['tracks']
            configure_ax_gtrack(self.ax_3, tracks)

            # FPS tracking
            current_time = time.time()
            self.frame_counter += 1
            if current_time - self.last_fps_time >= 1.0:  # Every 1 second
                self.fps = self.frame_counter / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
                self.frame_counter = 0

            # Update FPS text on the polar plot
            self.fps_text = self.ax.text(0.02, 1.02, f"FPS: {self.fps:.2f}", transform=self.ax.transAxes, fontsize=10,
                                         color='blue')

            # Update the figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            plt.pause(0.001)

        return Task.cont

def main(cfg_radar, cfg_gtrack, cfg_cfar, gtrack=True):
    q_main = Queue(maxsize=1)  # ❗️ Only keep latest

    producers = [Process(target=producer_real_time_1843, args=(q_main, cfg_radar, cfg_gtrack, cfg_cfar, gtrack), daemon=True)]
    consumers = [Process(target=consumer, args=(q_main, cfg_radar), daemon=True)]

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

