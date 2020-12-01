import rosbag
import argparse
import os
from src.evaluation import SegmentationGroundTruth
from src.evaluation import Segmentation
from src.evaluation import evaluate_instance_segmentation
from src.evaluation import evaluate_semantic_segmentation
from src.evaluation import print_pr_results
from src.evaluation import print_iou_list
from src.evaluation import calculate_precision_recall
from src.evaluation import plot_range_vs_accuracy
import re
import glob
import pypcd
import pprint
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.lib.recfunctions import unstructured_to_structured


def keyFunc(afilename):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", afilename))


def launchTrimmer(pcd_path, in_camera_view, r_mask, g_mask):
    # load cloud.pcd
    cloud = pypcd.PointCloud.from_path(pcd_path)
    pprint.pprint(cloud.get_metadata())

    # convert the structured numpy array to a ndarray
    trimmed_cloud = structured_to_unstructured(cloud.pc_data)

    # print the shape of the new array
    print(trimmed_cloud.shape)
    tp_mask = np.logical_and(r_mask, g_mask)
    fn_mask = np.logical_and(g_mask, np.logical_not(r_mask))
    fp_mask = np.logical_and(r_mask, np.logical_not(g_mask))
    trimmed_cloud = trimmed_cloud[in_camera_view]
    trimmed_cloud[:, 3] = trimmed_cloud[:, 3].astype("int32")
    trimmed_cloud[:, 4] = trimmed_cloud[:, 4].astype("int32")

    fn_cloud = trimmed_cloud[fn_mask]
    tp_cloud = trimmed_cloud[tp_mask]
    fp_cloud = trimmed_cloud[fp_mask]

    # this is necessary to distinguish between TP / FN / FP in the pcl viewer
    # the pcl viewer chooses a different color depending on the values in the label column
    # 0 is the background (FP), 1 stands for the class person (FN and TP)
    # to distinguish between FN and TP we add +1 to the TPs
    # this way we end up with 0 for background (FP), 1 for FN and 2 for TP
    # values in the column object have no effect on visualization whatsoever
    tp_cloud[:, 3] = tp_cloud[:, 3] + 1

    fn_tp_fp_cloud = np.concatenate((fn_cloud, tp_cloud, fp_cloud), axis=0)

    structured_fn_tp_fp_cloud = unstructured_to_structured(fn_tp_fp_cloud)
    pcd_cloud = from_array(structured_fn_tp_fp_cloud)

    # this can be visualized with the pcl_viewer tool as follows:
    # pcl_viewer -multiview 1 fn_tp_fp.pcd
    pypcd.save_point_cloud(pcd_cloud, 'fn_tp_fp.pcd')


def from_array(arr):
    """ create a PointCloud object from an array.
    """
    pc_data = arr.copy()
    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'label', 'object'],
          'size': [4, 4, 4, 4, 4],
          'count': [],
          'width': 0,
          'height': 1,
          'viewpoint': [0, 0, 0, 1, 0, 0, 0],
          'points': 0,
          'type': ['F', 'F', 'F', 'I', 'I'],
          'data': 'binary_compressed'}
    for field in md['fields']:
        md['count'].append(1)
    md['width'] = len(pc_data)
    md['points'] = len(pc_data)
    pc = pypcd.PointCloud(md, pc_data)
    return pc


if __name__ == '__main__':
    print("Start RSLS evaluation..")
    # read result bag
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-predictions",
        nargs="*",
        type=str
    )
    parser.add_argument(
        "-gt_path",
        nargs="*",
        type=str, help="path to gt files"
    )

    parser.add_argument(
        "-pcd_path",
        type=str, help="path to hitachi annotated pcd file to analyze",
        default=None
    )

    parser.add_argument(
        "-launch_trimmer",
        type=bool,
        default=False,
        help="calculate reduced pcd to fp and fn"
    )

    args = parser.parse_args()
    bag_path = args.predictions
    gt_path = args.gt_path
    pcd_path = args.pcd_path
    launch_trimmer = args.launch_trimmer

    in_camera_view = []
    prediction_list = []
    # bag_path is a list of paths pointing to distinct segmentation bags
    for bag in bag_path:
        inbag = rosbag.Bag(bag, 'r')
        for topic, msg, t in inbag.read_messages():
            prediction = Segmentation.parse_msg(msg)
            stamp = str(msg.header.stamp.secs) + '.' + "{:09d}".format(msg.header.stamp.nsecs)
            print(stamp)
            if pcd_path is not None:
                stamp = str(msg.header.stamp.secs) + '.' + "{:09d}".format(msg.header.stamp.nsecs)
                print(stamp)
                if stamp in pcd_path:
                    in_camera_view = prediction.in_camera_view
            prediction_list.append(prediction)

    gt_list = []
    # gt_path is a list of paths pointing to distinct gt .txt files
    for path in gt_path:
        for file in sorted(glob.glob(os.path.join(path, '*.txt')), key=keyFunc):
            print(file)
            gt = SegmentationGroundTruth.load_file(os.path.join(path, file))
            gt_list.append(gt)

    #### Evaluate class level semantic segmentation performance
    tp_totals, fp_totals, fn_totals, r_mask, g_mask = evaluate_semantic_segmentation(prediction_list, gt_list, pcd_path,
                                                                                     return_pr=True)
    iou_list = evaluate_semantic_segmentation(prediction_list, gt_list)
    print_iou_list(iou_list)
    print_pr_results(tp_totals, fp_totals, fn_totals)

    # visualize tp and fn for a specific .pcd annotated with the hitachi segmentation tool
    if launch_trimmer:
        launchTrimmer(pcd_path, in_camera_view, r_mask, g_mask)

    #### Evaluate object level semantic segmentation performance
    tp_totals, fp_totals, fn_totals = evaluate_instance_segmentation(prediction_list, gt_list)
    print_pr_results(tp_totals, fp_totals, fn_totals)
    precision, recall = calculate_precision_recall(tp_totals, fp_totals, fn_totals)
    plot_range_vs_accuracy(prediction_list, gt_list)

    print("Evaluation completed")
