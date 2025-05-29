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

        self.fig_3 = plt.figure(figsize=(6, 6))
        self.ax_3 = self.fig_3.add_subplot(111, projection='polar')
        self.im_3 = configure_ax_bf(self.ax_3, self.phi, self.r_idxs)

        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_text = self.ax.text(0.02, 1.02, "", transform=self.ax.transAxes, fontsize=10, color='blue')

        self.taskMgr.add(self.updateTask, "updateTask")

        self.x1, self.y1 = 0.0, 0.0
        self.x2, self.y2 = -10, 0.0

        self.x = np.arange(-60, 60, 1)
        self.y = np.arange(-60, 60, 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        x_flat = self.X.ravel()
        y_flat = self.Y.ravel()

        phi_flat = np.arctan2(y_flat, x_flat)
        r_flat = np.hypot(x_flat, y_flat)

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

            phi1 = np.arctan2((self.Y - self.y1).ravel(), (self.X - self.x1).ravel())
            r1 = np.hypot(self.X.ravel() - self.x1, self.Y.ravel() - self.y1)
            cart2pol1 = np.column_stack((phi1, r1))

            phi2 = np.arctan2((self.Y - self.y2).ravel(), (self.X - self.x2).ravel())
            r2 = np.hypot(self.X.ravel() - self.x2, self.Y.ravel() - self.y2)
            cart2pol2 = np.column_stack((phi2, r2))

            # build your fast polar→Cartesian interpolators
            interp1 = RegularGridInterpolator(
                (self.phi, self.r_idxs),  # φ axis, r axis
                bf_1,
                method='linear', bounds_error=False, fill_value=0
            )
            interp2 = RegularGridInterpolator(
                (self.phi, self.r_idxs),
                bf_2,
                method='linear', bounds_error=False, fill_value=0
            )

            # sample at every global (x,y) for each radar
            Z1 = interp1(cart2pol1).reshape(self.X.shape)
            Z2 = interp2(cart2pol2).reshape(self.X.shape)

            # Fuse
            Z_cart = (Z1 * Z2)

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

            # Build a Cartesian->grid interpolator once for the first radar
            interp_cart2pol = RegularGridInterpolator(
                (self.y, self.x),  # note order (row=y, col=x)
                Z1,
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
            Z1_polar = interp_cart2pol(pts_back).reshape(PHI.shape)

            # Build a Cartesian->grid interpolator once for the second radar
            interp_cart2pol = RegularGridInterpolator(
                (self.y, self.x),  # note order (row=y, col=x)
                Z2,
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
            Z2_polar = interp_cart2pol(pts_back).reshape(PHI.shape)

            # Normalize the output
            to_plot = np.abs(Z_polar)
            to_plot = to_plot
            to_plot /= np.max(to_plot)
            to_plot = to_plot ** 8

            to_plot_1 = np.abs(Z1_polar)
            to_plot_1 = to_plot_1
            to_plot_1 /= np.max(to_plot_1)
            to_plot_1 = to_plot_1 ** 8

            to_plot_2 = np.abs(Z2_polar)
            to_plot_2 = to_plot_2
            to_plot_2 /= np.max(to_plot_2)
            to_plot_2 = to_plot_2** 8

            # Update the beamforming plot
            self.im.set_array(to_plot.ravel())
            self.im_2.set_array(to_plot_1.ravel())
            self.im_3.set_array(to_plot_2.ravel())

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
            self.fig_3.canvas.draw_idle()

            QtWidgets.QApplication.processEvents()


            # Reset the message count
            self.msg_count.clear()

            plt.pause(0.001)

        return Task.cont

def main(cfg_radar, cfg_gtrack, cfg_cfar, gtrack=True):
    q_main_1 = Queue(maxsize=1)  # ❗️ Only keep latest
    q_main_2 = Queue(maxsize=1)

    producers = [
        Process(target=producer_real_time_1843, args=(q_main_2, cfg_radar, cfg_cfar, 4099, 5000, "192.168.33.32", "192.168.33.182"), daemon=True),
        Process(target=producer_real_time_1843, args=(q_main_1, cfg_radar, cfg_cfar, 4096, 4098, "192.168.33.30", "192.168.33.181"), daemon=True)]
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

