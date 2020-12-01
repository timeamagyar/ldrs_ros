from scipy.optimize import linear_sum_assignment
from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32
from decimal import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
GROUND_LEVEL = -1.6

CLASS_NAMES = ['BG', 'person']


class SegmentationGroundTruth(object):
    """
    Segmentation ground truth.
    """

    def __init__(self, instance_labels, class_labels, filename):
        self.instance_labels = np.array(instance_labels)
        self.class_labels = np.array(class_labels)
        self.filename = filename

    def filter(self, filter_array):
        self.instance_labels = self.instance_labels[filter_array]
        self.class_labels = self.class_labels[filter_array]

    ## class method used as an alternative constructor
    @classmethod
    def load_file(cls, filename):
        """
        Load ground truth from a .txt file with rows formatted as:
            instance_label class_label

        Instance and class labels are separated by a space.
        instance_label should be castable to an int,
        and class_label will be used as a string.

        Parameters
        ----------
        filename: str
            Name of file to load

        Returns
        -------
        SegmentationGroundTruth

        """
        with open(filename, "r") as loadfile:
            lines = loadfile.readlines()
        splitlines = [line.split(" ") for line in lines]
        instance_labels = [int(s[0]) for s in splitlines]
        class_labels = [s[1] for s in splitlines]
        class_labels = [l[:-1] if l.endswith('\n') else l for l in class_labels]
        return cls(instance_labels, class_labels, filename)

    @property
    def n_instances(self):
        return len(np.unique(self.instance_labels))


class Segmentation(object):
    """
    Segmentation result.
    """

    def __init__(self, instance_labels, class_labels, in_camera_view, points, stamp):
        self.instance_labels = instance_labels
        self.class_labels = class_labels
        self.in_camera_view = in_camera_view
        self.points = points
        self.stamp = stamp

    @classmethod
    def parse_msg(cls, msg):
        instance_labels = np.array(msg.instance_ids)
        class_labels = np.array(msg.class_names)
        in_camera_view = np.array(msg.in_camera_view)
        point_gen = read_points(msg.points)
        points = np.array([p for p in point_gen])
        stamp = str(msg.header.stamp.secs) + '.' + "{:09d}".format(msg.header.stamp.nsecs)
        return cls(instance_labels, class_labels, in_camera_view, points, stamp)


class InstanceSegmentationResults(object):

    def __init__(self, iou_threshold, n_classes):
        self.iou_threshold = iou_threshold
        self.tp_totals = [0 for i in range(n_classes)]
        self.fp_totals = [0 for i in range(n_classes)]
        self.fn_totals = [0 for i in range(n_classes)]


def evaluate_semantic_segmentation(results_list, gt_list,
                                   filename=None,
                                   range_limit=None,
                                   filter_ground=False,
                                   return_pr=False,
                                   return_pr_iu=False):
    """
    Evaluate labeling result as semantic segmentation (i.e. without considering object instances)

    Reports IoU over classes
    """

    # Define list of classes to evaluate
    coco_names = ['person']

    # Keep running total of intersection and union values, for each class
    i_totals = [0 for c in coco_names]
    u_totals = [0 for c in coco_names]
    fp_totals = [0 for c in coco_names]
    fn_totals = [0 for c in coco_names]
    r_totals = []
    g_totals = []


    for (results, gt) in zip(results_list, gt_list):
        # Get object class labels (not instance labels)
        results_class_labels = results.class_labels
        gt_class_labels = gt.class_labels
        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]

        if range_limit is not None:
            lidar_points = results.points
            ranges = np.linalg.norm(lidar_points, axis=1)
            in_range = ranges < range_limit
            results_class_labels = results_class_labels[in_range]
            gt_class_labels = gt_class_labels[in_range]

        if filter_ground:
            lidar_points = results.points
            if range_limit is not None:
                lidar_points = lidar_points[in_range, :]
            not_ground = lidar_points[:, 2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]

        # For each class C:
        for i in range(len(coco_names)):
            coco_class = coco_names[i]
            r = results_class_labels == coco_class
            g = gt_class_labels == coco_class
            if filename is not None:
                if results.stamp in filename:
                    r_totals = r
                    g_totals = g
            intersection = np.logical_and(r, g)
            union = np.logical_or(r, g)

            i_totals[i] += np.sum(intersection)
            u_totals[i] += np.sum(union)

            fp_totals[i] += np.sum(np.logical_and(r, np.logical_not(g)))
            fn_totals[i] += np.sum(np.logical_and(g, np.logical_not(r)))

    # true positives = intersection
    tp_totals = i_totals

    if return_pr:
        return tp_totals, fp_totals, fn_totals, r_totals, g_totals

    elif return_pr_iu:
        return tp_totals, fp_totals, fn_totals, i_totals, u_totals

    else:
        print(i_totals)
        print(u_totals)
        iou_list = [Decimal(i) / Decimal(u) for (i, u) in zip(i_totals, u_totals)]
        return iou_list


def evaluate_instance_segmentation(results_list,
                                   gt_list,
                                   iou_threshold=0.7,
                                   range_limit=None,
                                   filter_ground=False):
    """
    Evaluate labeling result as instance segmentation

    Reports IoU over instances

    Attributes
    ----------
    results_list: list
        List of LidarSegmentation
    gt_list: list
        List of LidarSegmentationGroundTruth
    iou_threshold: float
    range_limits: tuple, or None
        Specify range_limits to only look at objects at certain distances.
        Should contain two float values, e.g. (0,10) to look at objects
        from 0 to 10 meters away.
    """
    coco_names = ['person']
    # Keep running total of intersection and union values, for each class
    tp_totals = [0 for c in coco_names]
    fp_totals = [0 for c in coco_names]
    fn_totals = [0 for c in coco_names]

    for (results, gt) in zip(results_list, gt_list):
        # Get object class labels (not instance labels)
        results_class_labels = results.class_labels
        gt_class_labels = gt.class_labels

        results_instance_labels = results.instance_labels
        gt_instance_labels = gt.instance_labels

        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]
        if len(results_instance_labels) != len(gt_instance_labels):
            gt_instance_labels = gt_instance_labels[results.in_camera_view]

        # rt = np.vstack((results_instance_labels, results_class_labels)).T
        # np.savetxt('results_1576232556.545664072.txt', rt, fmt="%s")

        # gt = np.vstack((gt_instance_labels, gt_class_labels)).T
        # np.savetxt('gt_1576232556.545664072.txt', gt, fmt="%s")

        if range_limit is not None:
            lidar_points = results.points
            ranges = np.linalg.norm(lidar_points, axis=1)
            in_range = ranges < range_limit
            results_class_labels = results_class_labels[in_range]
            results_instance_labels = results_instance_labels[in_range]
            gt_class_labels = gt_class_labels[in_range]
            gt_instance_labels = gt_instance_labels[in_range]

        if filter_ground:
            lidar_points = results.points
            if range_limit is not None:
                lidar_points = lidar_points[in_range, :]
            not_ground = lidar_points[:, 2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            results_instance_labels = results_instance_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]
            gt_instance_labels = gt_instance_labels[not_ground]

        # For each class C:
        for i in range(len(coco_names)):
            coco_class = coco_names[i]

            r_is_class = results_class_labels == coco_class
            g_is_class = gt_class_labels == coco_class

            # Find instances of this class, in results and in ground truth
            r_instances = np.unique(results_instance_labels[r_is_class])
            g_instances = np.unique(gt_instance_labels[g_is_class])

            n_r = len(r_instances)
            n_g = len(g_instances)

            # Create IoU matrix
            # Is n by m, where n is the number of object instances in the segmentation results,
            # and m is the number of instances in the ground truth
            iou_matrix = np.zeros((n_r, n_g))

            for row in range(n_r):
                r_instance = results_instance_labels == r_instances[
                    row]  # Results instance number
                for col in range(n_g):
                    g_instance = gt_instance_labels == g_instances[
                        col]  # GT instance number
                    intersection = np.logical_and(r_instance, g_instance)
                    union = np.logical_or(r_instance, g_instance)
                    iou_matrix[row, col] = Decimal(np.sum(intersection)) / Decimal(np.sum(union))
            # row_ind(r_matching), col_ind(g_matching) : array
            # An array of row indices and one of corresponding column indices giving
            # the optimal assignment.
            r_matching, g_matching = linear_sum_assignment(
                cost_matrix=1 - iou_matrix)
            matching_matrix = np.zeros(iou_matrix.shape, dtype=int)

            tp_count = 0

            for (r, g) in zip(r_matching, g_matching):
                iou = iou_matrix[r, g]
                # print("Maximal matching: Matched results %d to GT %d, with iou %.3f" % (r,g,iou))
                if iou > iou_threshold:
                    matching_matrix[r, g] = 1
                    tp_count += 1

            # The number of all-zero rows in the matching matrix is the
            # number of false positives
            zero_rows = ~np.any(matching_matrix, axis=1)
            fp_count = np.sum(zero_rows)

            # The number of all-zero columns in the matching matrix is
            # the number of false negatives (undetected GT objects)
            zero_cols = ~np.any(matching_matrix, axis=0)
            fn_count = np.sum(zero_cols)

            tp_totals[i] += tp_count
            fp_totals[i] += fp_count
            fn_totals[i] += fn_count

    return tp_totals, fp_totals, fn_totals


# class level segmentation
def print_iou_list(iou_list, classes=('person',)):
    for (iou, name) in zip(iou_list, classes):
        print("IoU for class %s is %.3f" % (name, iou))


# class level segmentation
def print_pr_results(tp_totals, fp_totals, fn_totals,
                     classes=('person',)):
    for (tp, fp, fn, name) in zip(tp_totals, fp_totals, fn_totals,
                                  classes):
        tp_fp = tp + fp
        precision = Decimal(tp) / Decimal(tp_fp)
        tp_fn = tp + fn
        recall = Decimal(tp) / Decimal(tp_fn)
        print("For class %s, precision is %.3f and recall is %.3f" % (
            name, precision, recall))
        print("TP=%d, FP=%d, FN=%d" % (tp, fp, fn))


# instance level segmentation
def calculate_precision_recall(tp_totals, fp_totals, fn_totals):
    """
    Calculate list of precision and recall values from lists of true pos., 
    false pos., false neg. values.
    """
    precision = [Decimal(tp) / Decimal(tp + fp) if (tp + fp) > 0 else 0 for (tp, fp) in zip(tp_totals, fp_totals)]
    recall = [Decimal(tp) / Decimal(tp + fn) if (tp + fn) > 0 else 0 for (tp, fn) in zip(tp_totals, fn_totals)]
    return precision, recall


# instance level segmentation
def plot_range_vs_accuracy(results_list,
                           gt_list,
                           filter_ground=False):
    """
    Evaluate labeling result as instance segmentation

    Reports IoU over classes

    Attributes
    ----------
    results_list: list
        List of LidarSegmentation
    gt_list: list
        List of LidarSegmentationGroundTruth
    iou_threshold: float
    range_limits: tuple, or None
        Specify range_limits to only look at objects at certain distances.
        Should contain two float values, e.g. (0,10) to look at objects
        from 0 to 10 meters away.
    """
    # Define list of classes to evaluate
    coco_names = ['person']
    # Keep running total of intersection and union values, for each class
    tp_totals = [0 for c in coco_names]
    fp_totals = [0 for c in coco_names]
    fn_totals = [0 for c in coco_names]

    class_points = [np.empty((0, 2)) for c in coco_names]
    class_styles = ['.b', '^r']

    for (results, gt) in zip(results_list, gt_list):

        # Get object class labels (not instance labels)
        results_class_labels = results.class_labels
        gt_class_labels = gt.class_labels

        results_instance_labels = results.instance_labels
        gt_instance_labels = gt.instance_labels

        if len(results_class_labels) != len(gt_class_labels):
            gt_class_labels = gt_class_labels[results.in_camera_view]
        if len(results_instance_labels) != len(gt_instance_labels):
            gt_instance_labels = gt_instance_labels[results.in_camera_view]

        points = results.points
        ranges = np.linalg.norm(points, axis=1)

        if filter_ground:
            not_ground = points[:, 2] > GROUND_LEVEL
            results_class_labels = results_class_labels[not_ground]
            results_instance_labels = results_instance_labels[not_ground]
            gt_class_labels = gt_class_labels[not_ground]
            gt_instance_labels = gt_instance_labels[not_ground]
            ranges = ranges[not_ground]

        # Calculate mean range to each ground truth instance
        instance_ranges = [np.mean(ranges[gt_instance_labels == i]) for i in
                           range(1, gt.n_instances)]

        # For each class C:
        for i in range(len(coco_names)):
            coco_class = coco_names[i]
            r_is_class = results_class_labels == coco_class
            g_is_class = gt_class_labels == coco_class

            # Find instances of this class, in results and in ground truth
            r_instances = np.unique(results_instance_labels[r_is_class])
            g_instances = np.unique(gt_instance_labels[g_is_class])

            n_r = len(r_instances)
            n_g = len(g_instances)

            # Create IoU matrix
            # Is n by m, where n is the number of object instances in the segmentation results,
            # and m is the number of instances in the ground truth
            iou_matrix = np.zeros((n_r, n_g))

            for row in range(n_r):
                r_instance = results_instance_labels == r_instances[
                    row]  # Results instance number
                for col in range(n_g):
                    g_instance = gt_instance_labels == g_instances[
                        col]  # GT instance number
                    intersection = np.logical_and(r_instance, g_instance)
                    union = np.logical_or(r_instance, g_instance)
                    iou_matrix[row, col] = Decimal(np.sum(intersection)) / Decimal(np.sum(union))

            r_matching, g_matching = linear_sum_assignment(
                cost_matrix=1 - iou_matrix)

            for (r, g) in zip(r_matching, g_matching):
                iou = iou_matrix[r, g]
                # print("Maximal matching: Matched results %d to GT %d, with iou %.3f" % (r,g,iou))
                pt = np.array([instance_ranges[g], iou]).reshape((1, 2))
                print(iou)
                print(gt.filename)
                class_points[i] = np.append(class_points[i], pt, axis=0)


    for pts, style in zip(class_points, class_styles):
        plt.plot(pts[:, 0], pts[:, 1], style)
    plt.legend(coco_names)
   # plt.xlim(left=0)
    plt.xlabel("Range to object centroid [m]")
    plt.ylabel("IoU")
    plt.savefig("range_scatter.eps", bbox_inches='tight')
    plt.show()

