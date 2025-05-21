from mmwavecapture.radar import Radar
from mmwavecapture import dca1000

def main():

    # Initialize the DCA1000EVM
    dca = dca1000.DCA1000()
    dca.config.dca_ip = "192.168.33.180"
    dca.config.dca_config_port = 4096

    # Initialize the radar
    cfg_file = "../configs/profile_super.cfg"
    radar = Radar(
        config_port="/dev/tty.usbmodemR20910491",
        config_baudrate=115200,
        data_port="/dev/tty.usbmodemR20910494",
        data_baudrate=921600,
        config_filename=cfg_file,
        initialize_connection_and_radar=True,
        capture_frames=0,
    )

    print(radar.get_radar_status())
    radar.config()

    # Check DCA1000EVM connection
    if not dca.system_connection():
        raise RuntimeError(f"DCA1000EVM connection error at {4096}")

    # Initialize DCA1000EVM
    dca.reset_fpga()
    dca.config_fpga()
    dca.config_packet_delay()

    # Start DCA1000EVM
    dca.start_record()
    radar.start_sensor()

    socket_data = dca.get_socket_data("data")
    socket_config = dca.get_socket_data("config")

    socket_data.close()
    socket_config.close()

if __name__ == "__main__":
    main()