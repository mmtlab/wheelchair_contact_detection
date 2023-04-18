# -*- coding: utf-8 -*
"""
Created on Thu Feb 16 16:00:38 2023

@author: giamp
"""
import numpy as np
import pyrealsense2 as rs
import cv2
import basicmaster as bm
FRAME_NUMBER=180

# Definisci il percorso del file .bag
file_path =r"D:\01_raw\T001.bag"

# Inizializza un oggetto Pipeline
pipe = rs.pipeline()

# Crea un oggetto config per la Pipeline
cfg = rs.config()

# Imposta la configurazione per leggere dal file .bag
cfg.enable_device_from_file(cfg, file_path, repeat_playback = False)

# Avvia la Pipeline
profile = pipe.start(cfg)



aligned_stream = rs.align(rs.stream.color)
# Ciclo per iterare sui frame
for i in range(FRAME_NUMBER+1):
    # Prova ad ottenere il frame successivo
    try:
        frame= pipe.wait_for_frames()
    except RuntimeError:
        break
    if i==FRAME_NUMBER:
       
        # Ottieni il frame di profondità
       depth_frame = frame.get_depth_frame()

        # Ottieni il frame di colore
       color_frame = frame.get_color_frame()
        
       color_image_rgb = np.asanyarray(color_frame.get_data())
        
       depth_image = np.asanyarray(depth_frame.get_data())
       print(i)


#intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
#camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
     #                     [0, intrinsics.fy, intrinsics.ppy],
          #                [0, 0, 1]])

cv2.rectangle(color_image_rgb,(384,0),(510,128),(0,255,0),3)
bm.plots.pltsImg(color_image_rgb)
bm.plots.pltsImg(depth_image)