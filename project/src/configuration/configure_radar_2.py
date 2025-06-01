from mmwavecapture.radar import Radar
from mmwavecapture import dca1000_2


def main():
    """
    Main function to configure the DCA1000 EVM with another ip address and ports.
    """

    # Initialize the DCA1000EVM
    print("Starting radar...")
    dca = dca1000_2.DCA1000()

    # Check DCA1000EVM connection
    if not dca.system_connection():
        raise RuntimeError(f"DCA1000EVM connection error at {4096}")

    # Configure the DCA1000EVM
    status = dca.config_eeprom()
    print(status)
    print("config_eeprom done")
    dca.reset_fpga()

    # Get the socket data
    socket_data = dca.get_socket_data("data")
    socket_config = dca.get_socket_data("config")

    # Close the sockets
    socket_data.close()
    socket_config.close()

    print("AWR1843BOOST and DCA1000EVM configured successfully.")



if __name__ == "__main__":
    main()