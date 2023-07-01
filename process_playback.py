import cv2
import numpy as np
from pyk4a import PyK4APlayback
from helpers import colorize, convert_to_bgra_if_required

import pickle as pkl
import subprocess

name_prefix = "20230627"

read_filename = "C:/Users/Kin/Desktop/kinect_r/"+name_prefix+"_video.mkv"
save_folder = "C:/Users/Kin/Desktop/kinect_r"
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
 
# Read until video is completed
while True:
  try:
    capture = playback.get_next_capture()
    
    if capture.ir is not None:
      ir_raw = capture.ir
      ir_cropped = ir_raw
      # ir_cropped = ir_raw[crop_miny:crop_maxy, crop_minx:crop_maxx]
      
      ir_clipped = np.clip(ir_cropped, 0, 1000)
      # ir_clipped = np.clip(ir_cropped+100,160,5500) # from azure-acquire
      # ir_clipped = ((np.log(ir_clipped)-5)*70).astype(np.uint8) # from azure-acquire
      ir_normalized = cv2.normalize(ir_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
      ir_normalized = cv2.cvtColor(ir_normalized, cv2.COLOR_GRAY2BGR)
      cv2.imshow("IR", ir_normalized)
      out_ir.write(ir_normalized)
      
    if capture.depth is not None:
      depth_raw = capture.depth
      depth_cropped = depth_raw
      # depth_cropped = depth_raw[crop_miny:crop_maxy, crop_minx:crop_maxx]
      depth_clipped = np.clip(depth_cropped, 550, 750)
      depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
      cv2.imshow("Depth", depth_normalized)
      out_depth.write(depth_normalized)
      # depth_data.append(list(depth_cropped))
      
      # Save depth data for moseq2
      depth_save = depth_cropped
      depth_save = depth_save.astype(np.uint16)[None,:,:]
      
      frame_size = '{0:d}x{1:d}'.format(depth_save.shape[2], depth_save.shape[1])
      command = ['ffmpeg',
            '-y',
            '-loglevel', 'fatal',
            '-framerate', str(fps),
            '-f', 'rawvideo',
            '-s', frame_size,
            '-pix_fmt', 'gray16',
            '-i', '-',
            '-an',
            '-crf',str(10),
            '-vcodec', 'ffv1',
            '-preset', 'ultrafast',
            '-threads', str(1),
            '-slices', str('24'),
            '-slicecrc', str('1'),
            '-r', str(fps),
            filename]
    key = cv2.waitKey(10)
    if key != -1:
        break
  except EOFError:
      break
out_ir.release()
out_depth.release()
cv2.destroyAllWindows()

# print("Saving depth data...")
# depth_data = np.array(depth_data)
# with open("C:/Users/Kin/Desktop/kinect_r/"+name_prefix+"saved_depth.pkl", 'wb') as f:
#   pkl.dump(depth_data, f)
# print("Done.")