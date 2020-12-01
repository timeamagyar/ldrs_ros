import rosbag
import argparse
import os
import decimal


def write_bag(input_path, output_path, image_topic, lower_time_threshold, upper_time_threshold):
    inbag = rosbag.Bag(input_path, 'r')
    outbag = rosbag.Bag(output_path, 'w')
    print("Reading messages...")

    for topic, msg, t in inbag.read_messages():
        # Copy all messages in the input into the output bag and crop at a certain timestamp
        if topic == image_topic:
            stamp = str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs)
            print(stamp)

            if decimal.Decimal(stamp) < decimal.Decimal(lower_time_threshold):
                continue
            if stamp >= upper_time_threshold:
                break

        outbag.write(topic, msg, t)

    inbag.close()
    outbag.close()


if __name__ == '__main__':
    image_topic = "/device_0/sensor_0/Depth_0/image/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("-bagfile",
                        help="path to the bagfile to process")
    parser.add_argument("-lower_threshold",
                        help="path to the bagfile to process")
    parser.add_argument("-upper_threshold",
                        help="path to the bagfile to process")
    args = parser.parse_args()
    bag_path = args.bagfile
    lower_threshold = args.lower_threshold
    upper_threshold = args.upper_threshold
    write_bag(bag_path, "output.bag", image_topic, lower_threshold, upper_threshold)