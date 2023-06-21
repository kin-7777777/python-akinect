import numpy as np
import cv2
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord, ColorResolution, DepthMode, depth_image_to_color_camera_custom
import pickle
import time

file_path = 'C:/Users/Kin/Desktop'

res_color = (1920, 1080)
# res_color = (2048, 1536)
res_depth = (512, 512)
# res_depth = (640, 576)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc_color = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc_depth = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
out = cv2.VideoWriter(file_path+'/pyout.avi', fourcc_color, 30, res_color)
out_depth = cv2.VideoWriter(file_path+'/pyout_d.avi', fourcc_color, 30, res_color, False)

depth_data = []

print(f"Starting device #0")
config = Config(color_resolution=ColorResolution.RES_1080P, color_format=ImageFormat.COLOR_BGRA32, depth_mode=DepthMode.WFOV_2X2BINNED)
device = PyK4A(config=config, device_id=0)
device.start()

# print(f"Open record file {file_path}")
# record = PyK4ARecord(device=device, config=config, path=file_path+'/pyout.mkv')
# record.create()

try:
    print("Recording... Press CTRL-C to stop recording.")
    time_start = time.time()
    while True:
        capture = device.get_capture()
        cimg = cv2.resize(capture.color, res_color)[:, :, :-1]
        # Default transform
        # transformed_dimg = capture.transformed_depth # remember to access capture.depth first to initialize capture._depth
        # Custom transform
        if capture.depth is not None:
            transformed_dimg = depth_image_to_color_camera_custom(capture.depth, capture.depth, capture._calibration, capture.thread_safe, interp_nearest=False)[1]
        dimg_normalized = cv2.normalize(transformed_dimg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        cv2.imshow('Live Feed', dimg_normalized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(cimg)
        out_depth.write(dimg_normalized)
        # record.write_capture(capture)
        # try:
        #     depth_data = np.append(depth_data, transformed_dimg,)
        # except:
        #     depth_data = np.reshape(transformed_dimg, (res_color[1], res_color[0], 1))
        # depth_data.append(list(transformed_dimg))
except KeyboardInterrupt:
    print("CTRL-C pressed. Exiting.")
    print("Recording should be about "+str(time.time()-time_start)+" seconds.")
    out.release()
    out_depth.release()
depth_data = np.array(depth_data)
print('Done.')
# record.flush()
# record.close()
# print(f"{record.captures_count} frames written.")