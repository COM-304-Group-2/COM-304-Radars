capture_file = "continuous"

--TODO: edit this path
SAVE_DATA_PATH = "C:\\Users\\kresl\\Documents\\COM-304\\record\\" .. capture_file .. ".bin"

ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()