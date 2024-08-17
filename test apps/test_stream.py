#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:02:03 2024

@author: kerem
"""

import sounddevice as sd
import numpy as np
import time

FS = 48000
channel_count = 2
t = np.arange(start=1, stop=FS * 1 + 1) / FS
signal = np.atleast_2d(0.2 * np.sin(440 * 2 * np.pi * t)).T
current_frame = 0

def my_cb(outdata, frames, ctime, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(len(signal) - current_frame, frames)
    outdata[:chunksize, :channel_count] = \
        signal[current_frame:current_frame + chunksize].repeat(channel_count, axis=1)
    if chunksize < frames:
        outdata[chunksize:, :channel_count] = 0
        raise sd.CallbackStop()
    current_frame += chunksize

stream = sd.OutputStream(callback=my_cb, channels=channel_count)
stream.start()
time.sleep(3)

current_frame = 0
stream = sd.OutputStream(callback=my_cb, channels=channel_count)
stream.start()