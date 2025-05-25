import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import time
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type none')   # no native GL window

from PyQt5 import QtWidgets

from .prod_dca import producer_real_time_1843
from visualization.visualization import configure_ax_bf, configure_ax_db, configure_ax_gtrack


def consumer(q1, q2, cfg_radar):
    app = MyApp(q1, q2, cfg_radar)
    app.run()

class MyApp(ShowBase):
    def __init__(self, queue_1, queue_2, cfg_radar):
        ShowBase.__init__(self)
        self.q1 = queue_1
        self.q2 = queue_2
        self.latest_msg = {}
        self.msg_count = set()
        self.phi = cfg_radar["phi"]
        self.r_idxs = cfg_radar["range_idx"]

        #plt.ion() # Plus lent ??

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.im = configure_ax_bf(self.ax, self.phi, self.r_idxs)

        self.fig_2 = plt.figure(figsize=(6, 6))
        self.ax_2 = self.fig_2.add_subplot(111, projection='polar')
        self.im_2 = configure_ax_bf(self.ax_2, self.phi, self.r_idxs)

        #self.fig_3 = plt.figure(figsize=(8, 6), constrained_layout=True)
        #self.ax_3 = self.fig_3.add_subplot(111)

        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_text = self.ax.text(0.02, 1.02, "", transform=self.ax.transAxes, fontsize=10, color='blue')

        self.taskMgr.add(self.updateTask, "updateTask")

        # once, up front:
        self.x = np.arange(0, 150, 1)
        self.y = np.arange(0, 150, 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        # flatten
        x_flat = self.X.ravel()
        y_flat = self.Y.ravel()

        # polar coords of each Cartesian pixel
        phi_flat = np.arctan2(y_flat, x_flat)
        r_flat = np.hypot(x_flat, y_flat)

        # stack into (n_pts,2) for interpolation calls
        self.cart2pol = np.column_stack((phi_flat, r_flat))


    def updateTask(self, task):
        try:
            for pid, q in enumerate((self.q1, self.q2)):
                while not q.empty():
                    msg = q.get_nowait()
                    if msg[0] == 'bev':
                        self.latest_msg[pid] = msg[1]
                        self.msg_count.add(pid)

        except:
            pass


        # Check if we have received a new messages from both producers
        if self.msg_count == {0, 1}:
            # Unpack the latest message
            bf_1 = self.latest_msg[0]
            bf_2 = self.latest_msg[1]

            interp1 = RegularGridInterpolator(
                (self.phi, self.r_idxs),  # axis 0 = φ, axis 1 = r
                bf_1,
                method='linear', bounds_error=False, fill_value=0
            )
            interp2 = RegularGridInterpolator(
                (self.phi, self.r_idxs),
                bf_2,
                method='linear', bounds_error=False, fill_value=0
            )

            # sample both in one go
            Z1 = interp1(self.cart2pol).reshape(self.X.shape)
            Z2 = interp2(self.cart2pol).reshape(self.X.shape)

            # Fuse
            Z_cart = 0.5 * (Z1 + Z2)

            # Build a Cartesian->grid interpolator once for the fused map
            interp_cart2pol = RegularGridInterpolator(
                (self.y, self.x),  # note order (row=y, col=x)
                Z_cart,
                method='linear',
                bounds_error=False,
                fill_value=0
            )

            # Sample back on your original polar mesh
            PHI, R = np.meshgrid(self.phi, self.r_idxs, indexing='ij')
            pts_back = np.column_stack((
                (R * np.sin(PHI)).ravel(),  # y
                (R * np.cos(PHI)).ravel()  # x
            ))
            Z_polar = interp_cart2pol(pts_back).reshape(PHI.shape)



            # Update the beamforming plot
            self.im.set_array(Z_polar.ravel())
            self.im_2.set_array(bf_2.ravel())

            #self.ax_2.clear()
            #configure_ax_bf(self.ax_2)
            #plot_2d_heatmap(self.ax_2, bf_2, self.phi, self.r_idxs, vmin=0, vmax=0.1)

            # Update the gtrack plot
            #self.ax_3.clear()
            #tracks = gtrack['tracks']
            #configure_ax_gtrack(self.ax_3, tracks)

            # FPS tracking
            current_time = time.time()
            self.frame_counter += 1
            if current_time - self.last_fps_time >= 1.0:  # Every 1 second
                self.fps = self.frame_counter / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
                self.frame_counter = 0

            # Update FPS text on the polar plot
            self.fps_text.set_text(f"FPS: {self.fps:.2f}")

            # Update the figure
            self.fig.canvas.draw_idle()
            self.fig_2.canvas.draw_idle()

            QtWidgets.QApplication.processEvents()


            # Reset the message count
            self.msg_count.clear()

            plt.pause(0.001)

        return Task.cont

def main(cfg_radar, cfg_gtrack, cfg_cfar, gtrack=True):
    q_main_1 = Queue(maxsize=1)  # ❗️ Only keep latest
    q_main_2 = Queue(maxsize=1)

    producers = [
        Process(target=producer_real_time_1843, args=(q_main_1, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), daemon=True),
        Process(target=producer_real_time_1843, args=(q_main_2, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.181"), daemon=True)]
    consumers = [Process(target=consumer, args=(q_main_1, q_main_2, cfg_radar), daemon=True)]

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

