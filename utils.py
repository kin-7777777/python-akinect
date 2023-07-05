import subprocess
import numpy as np
import os
import json
from datetime import datetime

# Taken from Datta's azure-acquire repository.
def write_frames(filename, frames, threads=1, fps=30, crf=10,
                 pixel_format='gray16', codec='ffv1', close_pipe=True,
                 pipe=None, slices=24, slicecrc=1, frame_size=None, get_cmd=False):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
        filename (str): path to the file to write the frames to.
        frames (numpy.ndarray): frames to write to file
        threads (int, optional): the number of threads for multiprocessing. Defaults to 6.
        fps (int, optional): camera frame rate. Defaults to 30.
        crf (int, optional): constant rate factor for ffmpeg, a lower value leads to higher quality. Defaults to 10.
        pixel_format (str, optional): pixel format for ffmpeg. Defaults to 'gray8'.
        codec (str, optional): codec option for ffmpeg. Defaults to 'h264'.
        close_pipe (bool, optional): boolean flag for closing ffmpeg pipe. Defaults to True.
        pipe (subprocess.pipe, optional): ffmpeg pipe for writing the video. Defaults to None.
        slices (int, optional): number of slicing in parallel encoding. Defaults to 24.
        slicecrc (int, optional): protect slices with cyclic redundency check. Defaults to 1.
        frame_size (str, optional): size of the frame, ie 640x576. Defaults to None.
        get_cmd (bool, optional): boolean flag for outputtting ffmpeg command. Defaults to False.

    Returns:
        pipe (subprocess.pipe, optional): ffmpeg pipe for writing the video.
    """
 
    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])

    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-crf',str(crf),
               '-vcodec', codec,
               '-preset', 'ultrafast',
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(frames.shape[0]):
        pipe.stdin.write(frames[i,:,:].tobytes())

    if close_pipe:
        pipe.stdin.close()
        return None
    else:
        return pipe

# Adapted and modified from Datta's azure-acquire repository.
def write_images(image_queue, save_folder, filename):
    """
    start writing the images to videos

    Args:
        image_queue (Subprocess.queues.Queue): data stream from the camera
        filename_prefix (str): base directory where the videos are saved
    """
    pipe = None
    
    while True: 
        data = image_queue.get() 
        if len(data)==0: 
            pipe.stdin.close()
            break
        else:
            pipe = write_frames(os.path.join(save_folder, filename), data.astype(np.uint16)[None,:,:], codec='ffv1', close_pipe=False, pipe=pipe, pixel_format='gray16')

# Taken from Datta's azure-acquire repository.
def write_metadata(filename_prefix, subject_name, session_name, 
                   depth_resolution=[640, 576], little_endian=True, color_resolution=[640, 576]):
    """
    write recording metadata as json file.

    Args:
        filename_prefix (str): session directory to save recording metadata file in
        subject_name (str): subject name of the recording
        session_name (str): session name of the recording
        depth_resolution (list, optional): frame resolution of depth videos. Defaults to [640, 576].
        little_endian (bool, optional): boolean flag that indicates if depth data is little endian. Defaults to True.
        color_resolution (list, optional): frame resolution of ir video. Defaults to [640, 576].
    """
    
    # construct metadata dictionary
    metadata_dict = {"SubjectName": subject_name, 'SessionName': session_name,
                     "DepthResolution": depth_resolution, "IsLittleEndian": little_endian,
                     "ColorResolution": color_resolution, "StartTime": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}
    
    metadata_name = os.path.join(filename_prefix, 'metadata.json')

    with open(metadata_name, 'w') as output:
        json.dump(metadata_dict, output)