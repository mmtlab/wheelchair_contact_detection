# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:25:09 2023

@author: eferlius
"""

import rosbag
# please refer to http://docs.ros.org/en/lunar/api/rosbag/html/python/rosbag.bag.Bag-class.html
filename = r'D:\01_raw\T002.bag'

bag = rosbag.Bag(filename)

bag.get_message_count()
bag.get_start_time()
bag.get_end_time()
bag.get_type_and_topic_info()
counter = 0
for topic, msg, t in bag.read_messages():
    counter+=1
    if counter > 10:
        break
    print("{}: {} [{}]".format(topic,msg,t))
    
bag.close()


from rosbag.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# create reader instance and open for reading
with Reader(filename) as reader: # looks for .yaml file
    # # topic and msgtype information is available on .connections list
    # for connection in reader.connections:
    #     print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/imu_raw/Imu':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            print(msg.header.frame_id)

    # messages() accepts connection filters
    connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = deserialize_cdr(rawdata, connection.msgtype)
        print(msg.header.frame_id)