# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:50:05 2021

@author: kerem.basaran
"""

import time
import logging

logging.basicConfig(filename='speaker_test.log', encoding='utf-8', level=logging.INFO)

# User only changes these two rows:
# arguments are: year, month, day, hour, minute, 0, 0, 0, is_it_summertime
start_time = time.mktime((2021, 5, 21, 17, 0, 0, 0, 0, 1)) / 60**2  # hours
lost_time = 24 * 1.5  # hours

logging.info("\n\nStart logging")
logging.info("VSG3.5 test at 125°C")
logging.info(f"Test started at: {time.localtime(start_time*60**2)}")
logging.info(f"Lost time: {lost_time} hours")

while True:
    current_time = time.time() / 60**2  # hours
    try:
        time_stamp = time.strftime("%Y %b %d, %a, %H:%M")
        time_passed = current_time - start_time - lost_time
        register = f"\nLog time: {time_stamp}.\nTime tested: {int(time_passed):d} hours"
        print(register)
        logging.info(register)
        time.sleep(3)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Quitting...")
        logging.shutdown()
        break
