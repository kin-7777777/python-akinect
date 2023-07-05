import cv2
import numpy as np
from pyk4a import PyK4APlayback
from helpers import colorize, convert_to_bgra_if_required

import pickle as pkl
from multiprocessing import Process, Queue
import os

from utils import write_frames, write_images, write_metadata

name_prefix = "20230704"

# read_filename = "C:/Users/Kin/Desktop/kinect_r/"+name_prefix+"_video.mkv"
# read_filename = '/run/user/1000/gvfs/smb-share:server=staff3.ntu.edu.sg,share=lkc/research/Hiroshi Makino/User/KinOn/Data/misc/'+name_prefix+"/"+name_prefix+"_video.mkv"
read_filename = '/home/user/KO/data/processed_box/20230704_video.mkv'
# save_folder = "C:/Users/Kin/Desktop/kinect_r"
# save_folder = '/run/user/1000/gvfs/smb-share:server=staff3.ntu.edu.sg,share=lkc/research/Hiroshi Makino/User/KinOn/Data/misc/'+name_prefix
save_folder = '/home/user/KO/data/processed_box/'+name_prefix

res_color = (1920, 1080)
# res_depth = (512, 512)
res_depth = (640, 576)
res_cropped = res_depth
# res_cropped = (201, 140)
fps = 30

crop_minx = 170
crop_maxx = crop_minx + res_cropped[0]
crop_miny = 150
crop_maxy = crop_miny + res_cropped[1]


fourcc_color = cv2.VideoWriter_fourcc('M','J','P','G')
out_ir = cv2.VideoWriter(save_folder+'/'+name_prefix+'_pyout_ir.avi', fourcc_color, fps, res_cropped)
out_depth = cv2.VideoWriter(save_folder+'/'+name_prefix+'_pyout_d.avi', fourcc_color, fps, res_cropped, False)

playback = PyK4APlayback(read_filename)
playback.open()

depth_data = []

# write recording metadata (moseq2)
subject_name = "mouse"
write_metadata(save_folder, subject_name, "session_1")

# moseq2
device_timestamps = []
moseq2_data_queue = Queue()

vis_depth_queue = Queue()
write_process_vis_depth = Process(target=write_images, args=(vis_depth_queue, save_folder, name_prefix+"_vis_depth.avi"))
write_process_vis_depth.start()

vis_ir_queue = Queue()
write_process_vis_ir = Process(target=write_images, args=(vis_ir_queue, save_folder, name_prefix+"_vis_ir.avi"))
write_process_vis_ir.start()

write_process_moseq2 = Process(target=write_images, args=(moseq2_data_queue, save_folder, name_prefix+"_depth_data_moseq2.avi"))
write_process_moseq2.start()

frame_num = 0
frame_start = fps*60
frame_end = fps*60*20

# Read until video is completed
while True:
    try:
        capture = playback.get_next_capture()
        frame_num = frame_num + 1
        if frame_num >= frame_start:
            if capture.ir is not None:
                ir_raw = capture.ir
                ir_cropped = ir_raw
                # ir_cropped = ir_raw[crop_miny:crop_maxy, crop_minx:crop_maxx]
                
                ir_clipped = np.clip(ir_cropped, 0, 1000)
                # ir_clipped = np.clip(ir_cropped+100,160,5500) # from azure-acquire
                # ir_clipped = ((np.log(ir_clipped)-5)*70).astype(np.uint8) # from azure-acquire
                ir_normalized = cv2.normalize(ir_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                ir_normalized = cv2.cvtColor(ir_normalized, cv2.COLOR_GRAY2BGR)
                #   cv2.imshow("IR", ir_normalized)
                out_ir.write(ir_normalized)
                #   vis_ir_queue.put(vis_ir)
            
            if capture.depth is not None:
                depth_raw = capture.depth
                depth_cropped = depth_raw
                # depth_cropped = depth_raw[crop_miny:crop_maxy, crop_minx:crop_maxx]
                depth_clipped = np.clip(depth_cropped, 550, 750)
                depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                #   cv2.imshow("Depth", depth_normalized)
                out_depth.write(depth_normalized)
                # depth_data.append(list(depth_cropped))
                
                #   vis_depth_queue.put(vis_depth)
                
                # Save depth data for moseq2
                device_timestamps.append(capture.depth_timestamp_usec)
                depth_save = depth_cropped.astype(np.int16)
                moseq2_data_queue.put(depth_save)
                #   depth_save = depth_save.astype(np.uint16)[None,:,:]
                #   depth_pipe = write_frames(os.path.join(save_folder, name_prefix+'_depth_data_moseq2.avi'), depth_save, codec='ffv1', close_pipe=False, pipe=depth_pipe, pixel_format='gray16')
                
            key = cv2.waitKey(10)
            if key != -1:
                break
    except EOFError:
        break

# change from microsecond to millisecond
device_timestamps = np.array(device_timestamps)/1000
np.savetxt(os.path.join(save_folder, 'depth_ts.txt'),device_timestamps, fmt = '%f')

vis_ir_queue.put([])
write_process_vis_ir.join()
vis_depth_queue.put([])
write_process_vis_depth.join()
moseq2_data_queue.put([])
write_process_moseq2.join()
out_ir.release()
out_depth.release()
cv2.destroyAllWindows()

# print("Saving depth data...")
# depth_data = np.array(depth_data)
# with open("C:/Users/Kin/Desktop/kinect_r/"+name_prefix+"saved_depth.pkl", 'wb') as f:
#   pkl.dump(depth_data, f)
# print("Done.")