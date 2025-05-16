from streaming import (realtime_streaming_2)
import mmwavecapture.radar
import mmwavecapture.dca1000


def main():
    cfg_file = "configs/profile_1.cfg"
    radar = mmwavecapture.radar.Radar(
        config_port="/dev/tty.usbmodemR20910491",
        config_baudrate=115200,
        data_port="/dev/tty.usbmodemR20910494",
        data_baudrate=921600,
        config_filename=cfg_file,
        initialize_connection_and_radar=True,
        capture_frames=0,
    )

    realtime_streaming_2.main(
        exp_num=0,
        lua_file=cfg_file)

if __name__ == "__main__":
    main()