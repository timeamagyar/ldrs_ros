import rosbag
import argparse
import os
import glob
import re
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32
from sensor_msgs.msg import CameraInfo
from src.segmentation import PointCloudSegmentation
from src.detections import MaskRCNNDetections
from src.utils import Projection
from ldls_ros.msg import Segmentation

# COCO Class names
CLASS_NAMES = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def msg_to_detections(result_msg):
    """

    Parameters
    ----------
    result_msg: Result

    Returns
    -------

    """

    class_ids = np.array(result_msg.class_ids)
    scores = None  # not needed
    rois = None  # not needed
    bridge = CvBridge()

    if len(result_msg.masks) == 0:
        shape = (374, 1238)
        masks = np.empty((374, 1238, 0))
    else:
        masks_list = [bridge.imgmsg_to_cv2(m, 'mono8') for m in result_msg.masks]
        shape = masks_list[0].shape
        masks = np.stack(masks_list, axis=2)
        masks[masks == 255] = 1

    return MaskRCNNDetections(shape, rois, masks, class_ids, scores)

def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))

def write_bag(intrinsics_path, input_path, gt_path, output_path, mrcnn_results_topic, pc_topic):
    """
    Reads an input rosbag, and writes an output bag including all input bag
    messages as well as Mask-RCNN results, written to the following topics:
    mask_rcnn/result: mask_rcnn_ros.Result
    mask_rcnn/visualization: Image

    Parameters
    ----------
    input_path: str
    output_path: str
    image_topic: str

    Returns
    -------

    """
    intrinsics_bag = rosbag.Bag(intrinsics_path, 'r')
    data_bag = rosbag.Bag(input_path, 'r')
    outbag = rosbag.Bag(output_path, 'w')
    pc_list = []
    pc_headers = []
    pc_timestamp = []


    mrcnn_directory = 'mrcnn/'
    if not os.path.exists(mrcnn_directory):
        os.mkdir(mrcnn_directory)

    # Intrinsic camera parameters for raw distorted images
    # including the focal lengths (fx, fy) and principal points (ppx, ppy)
    intrinsics = []
    for topic, msg, t in intrinsics_bag.read_messages():
        print(msg)
        intrinsics = msg.K

    focal_lengths = [intrinsics[0], intrinsics[4]]
    print(focal_lengths)
    principal_points = [intrinsics[2], intrinsics[5]]
    print(principal_points)
    projection = Projection(focal_lengths, principal_points)

    # fill ground truth list from file
    #gt_list = ["1576232553.643975019", "1576232553.710680723"]

    gt_list = []
    for file in sorted(glob.glob(os.path.join(gt_path, '*.txt')), key=keyFunc):
        print(file)
        pcd_file = os.path.basename(file)
        base = os.path.splitext(pcd_file)[0]

        gt_list.append(base)

    print(gt_list)

    # Write all input messages to the output
    print("Reading messages...")
    for topic, msg, t in data_bag.read_messages():
        outbag.write(topic, msg, t)

    # Generate LDLS results
    for topic, msg, t in data_bag.read_messages(topics=[pc_topic]):
        point_gen = read_points(msg)
        points = np.array([p for p in point_gen])
        pc_list.append(points[:, 0:3])
        pc_headers.append(msg.header)
        pc_timestamp.append(t)
    print("Running LDLS...")
    point_cloud_seg = PointCloudSegmentation(projection)

    i = 0
    for topic, msg, t in data_bag.read_messages(topics=[mrcnn_results_topic]):
        if i % 50 == 0:
            print("Message %d..." % i)

        detections = msg_to_detections(msg)
        print("msg header")
        print(msg.header)
        # Get the class IDs, names, header from the MRCNN message
        class_ids = msg.class_ids
        class_names = list(msg.class_names)
        point_cloud = pc_list[i]
        print("i: ", i)
        print(len(pc_headers))
        header = pc_headers[i]
        ldls_res = point_cloud_seg.run(point_cloud, detections, save_all=False)

        print(header)


        stamp = str(msg.header.stamp.secs) + '.' + "{:09d}".format(msg.header.stamp.nsecs)
        point_cloud = point_cloud[ldls_res.in_camera_view, :]

        pc_msgs = []
        # Get segmented point cloud for each object instance
        labels = ldls_res.instance_labels()
        class_labels = ldls_res.class_labels()

        # only compute for evaluation candidates
        if stamp in gt_list:
            for inst in range(1, len(class_names) + 1):
                in_instance = labels == inst
                if np.any(in_instance):
                    inst_points = point_cloud[in_instance, :]
                    pc_msg = create_cloud_xyz32(header, inst_points)
                    pc_msgs.append(pc_msg)

            ldls_msg = Segmentation()
            ldls_msg.header = header
            ldls_msg.class_ids = ldls_res.class_labels().tolist()
            ldls_msg.class_names = np.array([CLASS_NAMES[j] for j in ldls_msg.class_ids])
            ldls_msg.in_camera_view = ldls_res.in_camera_view.tolist()
            ldls_msg.instance_ids = ldls_res.instance_labels()
            ldls_msg.object_points = pc_msgs
            ldls_msg.points = create_cloud_xyz32(header, ldls_res.points)
            outbag.write('/ldls/segmentation', ldls_msg, t)

        # this is mostly just for visualization
        foreground = point_cloud[class_labels != 0, :]
        foreground_msg = create_cloud_xyz32(header, foreground)
        outbag.write('/ldls/foreground', foreground_msg, t)

        i += 1

    intrinsics_bag.close()
    data_bag.close()
    outbag.close()


if __name__ == '__main__':
    # 2d segmentation results
    mrcnn_results_topic = '/object_detection/results'

    # pc2 in color frame
    pc_topic = '/object_detection/pc2'

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-intrinsics_bag",
                        help="path to the bagfile to process")

    parser.add_argument("-data_bag",
                        help="path to the bagfile to process")

    parser.add_argument(
        "-gt_path",
        type=str, help="path to gt data"
    )
    args = parser.parse_args()
    intrinsics_path = args.intrinsics_bag
    data_path = args.data_bag
    gt_path = args.gt_path

    if not os.path.exists(data_path):
        raise IOError("Bag file '%s' not found" % data_path)

    if not os.path.exists(gt_path):
        raise IOError("Ground truth file '%s' not found" % gt_path)

    out_name = data_path.split('.bag')[0] + '_ldls.bag'
    write_bag(intrinsics_path, data_path, gt_path, out_name, mrcnn_results_topic, pc_topic)
