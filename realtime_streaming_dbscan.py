import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import os
import time
from multiprocessing import Process, Queue
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import matplotlib.pyplot as plt
import numpy as np
from streaming.prod_dca import producer_real_time_1843

def consumer(q, index):
    app = MyApp(q)
    app.run()

class MyApp(ShowBase):
    def __init__(self, queue):
        ShowBase.__init__(self)
        self.q = queue
        self.latest_msg = None

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scat = self.ax.scatter([], [], c='blue', marker='o', alpha=0.7)
        self._configure_ax()

        self.taskMgr.add(self.updateTask, "updateTask")

    def _configure_ax(self):
        self.ax.set_title("2D Cluster View (DBSCAN)")
        self.ax.set_xlim(-100, 100)  # Adjust according to expected range
        self.ax.set_ylim(-100, 100)  # Adjust according to expected range
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")

    def updateTask(self, task):
        try:
            while not self.q.empty():
                msg = self.q.get_nowait()
                if msg[0] == "bev":
                    self.latest_msg = msg[1]
        except:
            pass

        if self.latest_msg:
            clusters = self.latest_msg
            self.ax.clear()
            self._configure_ax()

            for label, cluster_points, centroid in clusters:
                if label == -1:
                    continue  # Ignore noise
                # Plot each cluster
                self.ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, label=f'Person {label+1}', alpha=0.7)
                # Plot the centroid
                self.ax.scatter(centroid[0], centroid[1], s=100, c='red', marker='x')

            self.ax.legend()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

        return Task.cont

def main(exp_num, lua_file):
    q_main = Queue(maxsize=1)

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

if __name__ == '__main__':
    home_dir = r'C:\\Users\\kresl\\Documents\\COM-304'
    config_lua_script = os.path.join(home_dir, r'scripts\\1843_config_streaming.lua')
    main(0, config_lua_script)
