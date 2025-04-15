import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import os
import time
import signal
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from streaming.prod_dca import producer_real_time_1843

def consumer(q, index):
    app = MyApp(q)
    app.run()

class MyApp(ShowBase):
    def __init__(self, queue):
        ShowBase.__init__(self)
        self.q = queue

        self.rfft_size = 512
        self.rfft_range = [0, 1]
        self.rfft_x_data = np.arange(self.rfft_size)
        self.rfft_y_data = np.zeros_like(self.rfft_x_data)

        plt.ion()
        self.fig, (self.ax_rfft, self.ax_ra) = plt.subplots(1, 2, figsize=(12, 5))

        # Range FFT plot
        (self.line_rfft,) = self.ax_rfft.plot(self.rfft_x_data, self.rfft_y_data)
        self.ax_rfft.set_title("Range FFT")
        self.ax_rfft.set_ylim(self.rfft_range)

        # Range-Angle map
        self.range_angle_data = np.zeros((4, 256))
        self.ra_img = self.ax_ra.imshow(self.range_angle_data, aspect='auto', cmap='jet', origin='lower',
                                        extent=[0, 256, -2, 2])
        self.ax_ra.set_title("Radar View (Range-Angle)")
        self.ax_ra.set_xlabel("Range Bin")
        self.ax_ra.set_ylabel("Angle Bin (normalized)")

        self.taskMgr.add(self.updateDataTask, "updateDataTask")
        self.taskMgr.add(self.updatePlotTask, "updatePlotTask")

    def updateDataTask(self, task):
        try:
            while not self.q.empty():
                msg = self.q.get_nowait()
                if msg[0] == "rfft":
                    self.rfft_y_data = msg[1]
                elif msg[0] == "range_angle":
                    self.range_angle_data = msg[1]
        except:
            pass
        return Task.cont

    def updatePlotTask(self, task):
        self.line_rfft.set_ydata(self.rfft_y_data)
        self.ax_rfft.set_ylim([np.min(self.rfft_y_data) - 1, np.max(self.rfft_y_data) + 1])

        self.ra_img.set_data(self.range_angle_data)
        self.ra_img.set_clim(vmin=np.min(self.range_angle_data), vmax=np.max(self.range_angle_data))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return Task.cont


def main(exp_num, lua_file):
    from multiprocessing import Queue
    q_main = Queue(maxsize=20)

    producers = [Process(target=producer_real_time_1843, args=(q_main, 0, lua_file), daemon=True)]
    consumers = [Process(target=consumer, args=(q_main, 0), daemon=True)]

    for p in producers: p.start()
    for c in consumers: c.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Ctrl+C received. Terminating processes...")
        for p in producers: p.terminate()
        for c in consumers: c.terminate()
        for p in producers: p.join()
        for c in consumers: c.join()
        print("✅ Shutdown complete.")

if __name__ == '__main__':
    home_dir = r'C:\\Users\\kresl\\Documents\\COM-304'
    config_lua_script = os.path.join(home_dir, r'scripts\1843_config_streaming.lua')
    main(0, config_lua_script)
