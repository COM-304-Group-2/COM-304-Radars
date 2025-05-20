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

import numpy as np
from .prod_dca_2 import producer_real_time_1843

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
        self.phi = np.deg2rad(np.arange(0, 180, 1))
        self.r_idxs = np.arange(0, 150)
        self.bev_map = np.zeros((len(self.phi), len(self.r_idxs)))
        self.phi_db = np.arange(0, 180, 1) * np.pi / 180

        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self._configure_ax()

        # Build full coordinate grid
        self.phi_rad_2d, self.r_idxs_2d = np.meshgrid(self.phi_db, self.r_idxs, indexing='ij')  # shape: (180, 140)

        #self.x_coords_m = np.cos(self.phi_rad_2d) * self.r_idxs_2d  # shape: (180, 140)
        #self.z_coords_m = np.sin(self.phi_rad_2d) * self.r_idxs_2d # shape: (180, 140)

        #self.x_coords_m = np.linspace(-180, 180, 1)
        #self.z_coords_m = np.linspace(0, 140, 1)

        #self.fig_2 = plt.figure(figsize=(6, 4))
        #self.ax_2 = self.fig_2.add_subplot(111)
        #self._configure_ax_2()

        self.fig_3 = plt.figure(figsize=(6, 6))
        self.ax_3 = self.fig_3.add_subplot(111)

        #self.db = None
        #self.points_thresh = None

        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()

        self.fps_text = self.ax.text(0.02, 1.02, "", transform=self.ax.transAxes, fontsize=10, color='blue')

        plt.show(block=False)

        self.taskMgr.add(self.updateTask, "updateTask")

        self.detection = None
        self.tracks = None

    def _configure_ax(self):
        self.ax.set_theta_zero_location('E')
        self.ax.set_theta_direction(1)
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(180)
        self.ax.set_title("Bird Eye View (Top View)")

    def _configure_ax_2(self):
        self.ax_2.set_xlabel("X")
        self.ax_2.set_ylabel("Y")
        self.ax_2.set_title("DBSCAN Clustering on Full Heatmap")

    def _configure_ax_3(self):
        self.ax_3.set_xlim(-70, 70)
        self.ax_3.set_ylim(0, 150)

        #self.ax_3.set_aspect('equal', adjustable='box')  # keep units equal
        self.ax_3.autoscale(enable=False)

        for tr in self.tracks:
            if tr['status'] != 'ACTIVE':
                continue

            x, y = tr['pos']
            vx, vy = tr['vel']
            uid = tr['uid']
            confidence = tr.get('confidence', 1.0)

            # draw the position
            self.ax_3.scatter(x, y, s=100 * confidence, edgecolors='k', facecolors='none')
            # draw an arrow showing velocity
            self.ax_3.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, width=0.005)

            # label with the track ID
            self.ax_3.text(x, y, f"{uid}", fontsize=12, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

        self.ax_3.set_xlabel("X position (m)")
        self.ax_3.set_ylabel("Y position (m)")
        self.ax_3.set_title("GTRACK 2D Tracks (size ∝ confidence)")
        #self.ax_3.axis('equal')
        self.ax_3.grid(True)

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

            self.phi, self.r_idxs, self.bev_map, self.detection = self.latest_msg
            self.ax.clear()
            self._configure_ax()
            plot_2d_heatmap(self.ax, self.bev_map, self.phi, self.r_idxs, vmin=0, vmax=0.1)

            #self.ax_2.clear()
            #self._configure_ax_2()
            #self.ax_2.imshow(self.bev_map.T, extent=[self.x_coords_m.min(), self.x_coords_m.max(), self.z_coords_m.min(), self.z_coords_m.max()],
             #          origin='lower', aspect='auto', cmap='hot')

            #labels = self.db.labels_
            # Plot clusters
            #for label in np.unique(labels):
             #   if label == -1:
              #      continue  # noise
               # cluster_pts = self.points_thresh[labels == label]
                #self.ax_2.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=30, label=f'Person {label + 1}', alpha=0.7)

            #self.ax_2.legend()
            #self.ax_2.grid(True)

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

            self.ax_3.clear()
            self.tracks = self.detection['tracks']
            self._configure_ax_3()

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

