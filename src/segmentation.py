import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.preprocessing import normalize
from numba import cuda
import math
import numba
import cupy as cp
import scipy
import time

NO_LABEL = -1

class PointCloudSegmentationResult(object):
    """
    Output from point cloud segmentation
    
    """

    def __init__(self, points, projected, label_likelihoods,
                 class_ids, in_camera_view, initial_labels):
        self.points = points  # all points: xyz
        self.projected = projected
        self.label_likelihoods = label_likelihoods  # n_iterations by n_points by n_instances
        self.class_ids = class_ids  # length n_instances (NOT counting background)
        self.in_camera_view = in_camera_view  # indices of points that are in the view of the color camera
        self.initial_labels = initial_labels

    @classmethod
    def load_file(cls, filename):
        """
        
        Parameters
        ----------
        filename: str
            Name of file to load

        Returns
        -------
        PointCloudSegmentationResult

        """
        if not filename.endswith(".npz"):
            filename += ".npz"
        with open(filename, "rb") as loadfile:
            npzfile = np.load(loadfile)
            results = cls(points=npzfile["points"],
                          projected=npzfile["projected"],
                          label_likelihoods=npzfile["label_likelihoods"],
                          class_ids=npzfile["class_ids"],
                          in_camera_view=npzfile["in_camera_view"],
                          initial_labels=npzfile["initial_labels"])
        return results

    def to_file(self, filename):
        """
        
        Parameters
        ----------
        filename: str

        Returns
        -------
        None

        """
        with open(filename, "wb") as savefile:
            np.savez_compressed(savefile, points=self.points,
                                projected=self.projected,
                                label_likelihoods=self.label_likelihoods,
                                class_ids=self.class_ids,
                                in_camera_view=self.in_camera_view,
                                initial_labels=self.initial_labels)

    def point_confidence(self, iter=-1):
        """
        Returns confidence of the instance label prediction for each point.
        Confidence is calculated using softmax function.
        
        Parameters
        ----------
        iter: int
            The iteration at which to compute results.

        Returns
        -------
        numpy.ndarray
            n_points-element vector

        """
        L = self.label_likelihoods[iter, :, :]
        labels = self.instance_labels(iter)
        L = normalize(L, axis=1)
        exp_L = np.exp(L)

    def instance_confidence(self, iter=-1):
        pass

    def instance_labels(self, iter=-1, remove_outliers=False):
        L = self.label_likelihoods[iter, :, :]
        labels = np.argmax(L, axis=1)
        if remove_outliers:
            labels = self.remove_outliers_depth(labels)
        return labels

    def class_labels(self, iter=-1):
        # Instance 0 is background (class 0)
        instance_labels = self.instance_labels(iter=iter) - 1
        if len(self.class_ids) == 0:  # No objects, return all zeros
            return np.zeros(instance_labels.shape, dtype=int)
        labels = self.class_ids[instance_labels]
        labels[instance_labels == -1] = 0
        return labels

    @property
    def n_iters(self):
        return self.label_likelihoods.shape[0]

    # @property
    # def instance_labels(self):
    #     return self.instance_labels(iter=-1)
    #
    # @property
    # def class_labels(self):
    #     return self.class_labels(iter=-1)

    def remove_outliers_depth(self, labels, threshold=1.0):
        """
        Simple outlier removal method to use as a baseline
        Parameters
        ----------
        labels
        threshold

        Returns
        -------

        """
        # Remove outliers based on median depth of each cluster
        # threshold should change based on class (i.e. larger for trucks than for people)
        point_cloud = self.points
        point_x = point_cloud[:, 0]
        for instance_label in np.unique(labels):
            # Ignore the background, and unlabeled points
            if instance_label == NO_LABEL or instance_label == 0:
                continue
            in_mask = labels == instance_label
            instance_points_x = point_x[in_mask]
            # Ignore masks that have no associated labelled points
            if len(instance_points_x) == 0:
                continue
            median_depth = np.median(instance_points_x)
            is_outlier = (np.abs(point_x - median_depth) > threshold).flatten()
            labels[in_mask & is_outlier] = 0  # set to background
        return labels


@numba.njit
def get_pixel_indices(x, y, indices_matrix, kernel_size):
    row = int(np.floor(y))
    col = int(np.floor(x))
    indices = []
    radius = kernel_size // 2
    for r in range(row - radius - 1, row + radius):
        if r < 0 or r > indices_matrix.shape[0]:
            continue
        for c in range(col - radius - 1, col + radius):
            if c < 0 or c > indices_matrix.shape[1]:
                continue
            indices.append(indices_matrix[r, c])
    return indices


@cuda.jit
def connect_pc_to_pixels(point_cloud, projected, pixel_indices_matrix, kernel_size,
                         weight, out_rows, out_cols, out_weight):
    """
    Compute connections from points to image pixels on the GPU.
    Parallelizes over points.

    All input arrays should be on the GPU (cupy arrays).

    Parameters
    ----------
    points: ndarray
        n_points by 3
        3D points.
    projected: ndarray
        n_points by 2
        2D image pixel coordinate projections of the points
    pixel_indices_matrix: ndarray
        n_rows by n_cols (i.e. shape of the 2D RGB image)
        Matrix of pixel indices.
        Can get this in numpy with:
        np.arange(n_rows*n_cols).reshape((n_rows, n_cols)).astype(int)
    kernel_size: int
        Lidar points are connected to all pixels within a box of this size,
        around the point's 2D projection.
        So if kernel_size=5, each point is connected to the 25 pixels
        around the point's projected 2D location.
    weight: float
        Constant weight value for all point-to-pixel connections in the graph.
    out_rows: ndarray
        n_points by (kernel_size * kernel_size)
        Output array. Row indices (i.e. point indices) will be saved in this array.
        Should be initialized to have all entries be -1 (or some other
        negative value). Invalid values (from when points connect to
        some pixel coordinates that are outside of the image) will be left as
        the initial value.
    out_cols: ndarray
        n_points by (kernel_size * kernel_size)
        Output array. Column indices (i.e. pixel indcies) will be saved in
        this array.
    out_weight: ndarray
        n_points by (kernel_size * kernel_size)
        Output array. Entries for valid point-to-pixel connections will be set
        to the "weight" argument value.

    Returns
    -------
    None

    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n_points = point_cloud.shape[0]

    for i in range(start, n_points, stride):
        x = projected[i, 0]
        y = projected[i, 1]
        row = int(math.floor(y))
        col = int(math.floor(x))
        radius = kernel_size // 2
        j = 0  # index into the kernel (goes from 0 to kernel_size**2)
        for r in range(row - radius - 1, row + radius):
            if r < 0 or r > pixel_indices_matrix.shape[0]:
                continue
            for c in range(col - radius - 1, col + radius):
                if c < 0 or c > pixel_indices_matrix.shape[1]:
                    continue
                pixel = pixel_indices_matrix[r, c]
                out_rows[i, j] = i
                out_cols[i, j] = pixel
                out_weight[i, j] = weight
                j += 1


@numba.njit()
def row_normalize(row_indices, d, n_points):
    """
    Used to row-normalize a coordinate format-specified sparse matrix.
    Ignores rows past n_points.

    Parameters
    ----------
    row_indices
    d
    n_points

    Returns
    -------

    """
    d_norm = np.empty(d.shape)
    row_sums = [0.0 for i in range(n_points)]
    for i in range(len(row_indices)):
        row = row_indices[i]
        if row < n_points:
            s = row_sums[row_indices[i]]
            row_sums[row_indices[i]] = s + d[i]

    for i in range(len(d)):
        row = row_indices[i]
        if row < n_points:
            d_norm[i] = d[i] / row_sums[row_indices[i]]
    return d_norm


class PointCloudSegmentation(object):
    """
    Class for performing segmentation of point clouds.
    """

    def __init__(self, projection, num_iters=-1, num_neighbors=10,
                 distance_scale=1.0,
                 mask_shrink=0.5, mask_dilate=2.0, outlier_removal=True,
                 pixel_to_pc_kernel_size=5,
                 pixel_to_pc_weight=0.001):
        """

        Parameters
        ----------
        projection
        num_iters: int
            If set to <= 0, will iterate until convergence (slower)
        num_neighbors
        mask_shrink
        """
        self.projection = projection
        self.num_iters = num_iters
        self.num_neighbors = num_neighbors
        self.distance_scale = distance_scale
        self.mask_shrink = mask_shrink
        self.mask_dilate = mask_dilate
        self.outlier_removal = outlier_removal
        self.pixel_to_pc_kernel_size = pixel_to_pc_kernel_size
        self.pixel_to_pc_weight = pixel_to_pc_weight

    def project_points(self, point_cloud):
        return self.projection.project(point_cloud)

    def get_in_view(self, point_cloud, projected, img_rows, img_cols):
        in_frame_x = np.logical_and(projected[:, 0] > 0,
                                    projected[:, 0] < img_cols - 1)
        in_frame_y = np.logical_and(projected[:, 1] > 0,
                                    projected[:, 1] < img_rows - 1)

        projected_in_frame = np.logical_and(in_frame_x, in_frame_y)
        # point is in view if it projects inside the image
        in_view = np.logical_and(~np.isnan(point_cloud[:, 0]), ~np.isnan(point_cloud[:, 1]), projected_in_frame)
        # check which points project to within the image bounds
        return in_view

    def create_graph(self, point_cloud, projected, n_rows, n_cols):
        """

        Parameters
        ----------
        point_cloud: ndarray
            N by 3
            3D points (assumed to only be those in camera view)
        projected: ndarray
            N by 2
            points projected into 2D image pixel coordinates
        n_rows: int
            Number of rows in the image
        n_cols: int
            Number of columns in the image

        Returns
        -------
        cupy.sparse.csr_matrix
            Sparse graph of size (N+P) by (N+P), where N is number of points
            and P is number of image pixels.
            The upper-left N by N quadrant is the KNN graph of points.
            The upper-right N by P quadrant is connections from the pixels
            to points.
            All entries in the bottom P by (N+P) half of the matrix are 0.
        """
        n_points = point_cloud.shape[0]
        n_pixels = n_rows * n_cols
        # Step 1: Create KNN graph of points only
        distances, neighbors = self.point_nearest_neighbors(point_cloud)

        # COO matrix initialization
        d = np.exp(-(distances ** 2) / (self.distance_scale ** 2)).flatten()
        row_indices = np.indices(distances.shape)[0].flatten()
        col_indices = neighbors.flatten()

        # Step 3: Find where points project to in the image
        # This is parallelized on the GPU for fast performance
        # This step of graph creation takes about 5-10 ms
        pixel_indices_matrix = np.arange(n_rows * n_cols).reshape(
            (n_rows, n_cols)).astype(int)

        # CUDA config
        blocks = 30
        threads = 256

        # outputs for point-to-pixel connections
        pp_rows_out = cp.full((n_points, self.pixel_to_pc_kernel_size ** 2),
                              -1, dtype=int)
        pp_cols_out = cp.full((n_points, self.pixel_to_pc_kernel_size ** 2),
                              -1, dtype=int)
        pp_d_out = cp.full((n_points, self.pixel_to_pc_kernel_size ** 2),
                           -1, dtype=cp.float32)

        connect_pc_to_pixels[blocks, threads](point_cloud, projected,
                                              pixel_indices_matrix,
                                              self.pixel_to_pc_kernel_size,
                                              self.pixel_to_pc_weight,
                                              pp_rows_out, pp_cols_out,
                                              pp_d_out)
        cuda.synchronize()

        valid = (pp_rows_out.ravel() >= 0).get()

        pp_rows = pp_rows_out.get().flatten()[valid]
        pp_cols = pp_cols_out.get().flatten()[valid] + n_points
        pp_d = pp_d_out.get().flatten()[valid]

        row_indices = np.concatenate([row_indices, pp_rows])
        col_indices = np.concatenate([col_indices, pp_cols])
        d = np.concatenate([d, pp_d])

        # Row-normalize
        d = row_normalize(row_indices, d, n_points)

        # Set bottom right quadrant to identity
        eye_indices = np.arange(n_points, n_points + n_pixels)
        row_indices = np.concatenate([row_indices, eye_indices])
        col_indices = np.concatenate([col_indices, eye_indices])
        old_shape = d.shape
        # d = np.concatenate([d, np.ones(n_pixels, dtype=d.dtype)])
        ones = np.ones(n_pixels, dtype=d.dtype)
        d = np.concatenate([d, np.ones(n_pixels, dtype=d.dtype)])

        same = np.logical_and((row_indices == col_indices),
                              (row_indices > n_points))


        S = scipy.sparse.coo_matrix((d, (row_indices, col_indices)),
                                    shape=(
                                        n_points + n_pixels, n_points + n_pixels))
        return cp.sparse.csr_matrix(S)

    def point_nearest_neighbors(self, point_cloud):
        n_points = point_cloud.shape[0]

        # Find nearest neighbors between points
        kdt = KDTree(point_cloud)

        # do KDT query with k+1 to account for point being own nearest neighbor
        distances, neighbors = kdt.query(point_cloud, k=self.num_neighbors + 1)

        return distances, neighbors

    def class_mass_normalize(self, label_likelihoods, detections):
        class_masses = [np.sum(detections.masks[:, :, i])
                        for i in range(len(detections))]
        # insert BG mass
        bg_mass = detections.masks.shape[0] * detections.masks.shape[
            1] - np.sum(class_masses)
        class_masses.insert(0, bg_mass)
        class_proportions = class_masses / np.sum(class_masses)
        cmn_likelihoods = np.divide(label_likelihoods,
                                    np.sum(label_likelihoods, axis=0))
        cmn_likelihoods = np.multiply(cmn_likelihoods, class_proportions)
        return cmn_likelihoods

    def remove_outliers(self, final_label_likelihoods, G):
        final_lh = final_label_likelihoods
        n_points, n_objs = final_lh.shape
        # graph = get_upper_left(G, n_points)
        graph = G[:n_points, :].tocsc()[:, :n_points].tocsr().get()
        inst_labels = np.argmax(final_lh, axis=1)
        point_indices = np.arange(n_points)
        for i in range(1, n_objs):  # start at 1 to skip background
            # find which points are labelled as object i
            labelled = inst_labels == i
            object_point_indices = point_indices[labelled]
            # Skip object i if no points have label i
            if np.sum(labelled) == 0:
                continue
            # get portion of graph for object i points
            subgraph = graph[labelled, :]
            subgraph = subgraph[:, labelled]

            # find connected components
            n_comp, comp_labels = scipy.sparse.csgraph.connected_components(
                subgraph,
                directed=False)

            _, comp_counts = np.unique(comp_labels, return_counts=True)
            largest_comp = np.argmax(comp_counts)
            outliers = comp_labels != largest_comp
            outlier_indices = object_point_indices[outliers]
            final_lh[outlier_indices,
            1:] = 0  # set outlier points to background
        return final_lh

    def run(self, point_cloud, detections, max_iters=200, device=0, save_all=True):
        with cp.cuda.Device(device):
            # Project points into 2D
            projected = self.project_points(point_cloud)
            n_rows = detections.masks.shape[0]
            n_cols = detections.masks.shape[1]
            n_pixels = n_rows * n_cols
            in_view = self.get_in_view(point_cloud, projected, n_rows, n_cols)
            point_cloud = point_cloud[in_view, :]
            n_points = point_cloud.shape[0]

            # Create initial label matrix by reshaping masks into vectors
            n_instances = len(detections)
            # Note that background is an extra instance
            if detections.masks.size > 0:  # handle case where no objects detected
                pixel_labels = detections.masks.reshape((-1, n_instances))
                pixel_labels = np.concatenate(
                    [detections.get_background().reshape((n_pixels, 1)),
                     pixel_labels], axis=1)
            else:
                pixel_labels = detections.get_background().reshape((n_pixels, 1))

            # Append initial zero labels for points
            labels = np.zeros((n_points + n_pixels, n_instances + 1))
            initial_point_labels = np.zeros((n_points, n_instances + 1))
            labels[n_points:, :] = pixel_labels
            # Move labels to device
            Y_gpu = cp.array(labels)

            # Create graph on GPU
            # This is a (n_points + n_pixels) by (n_points + n_pixels) matrix
            G_gpu = self.create_graph(point_cloud, projected[in_view, :],
                                      n_rows=detections.masks.shape[0],
                                      n_cols=detections.masks.shape[1])

            if save_all:
                all_label_likelihoods = np.empty(
                    (max_iters + 1, n_points, n_instances + 1))
            else:
                all_label_likelihoods = np.empty(
                    (2, n_points, n_instances + 1))

            for i in range(max_iters):
                Y_new = G_gpu.dot(Y_gpu)
                Y_gpu = Y_new
                if save_all:
                    all_label_likelihoods[i + 1, :, :] = cp.asnumpy(
                        Y_gpu[:n_points, :])
            if not save_all:
                all_label_likelihoods[-1, :, :] = cp.asnumpy(
                    Y_gpu[:n_points, :])

            # Remove outliers
            if self.outlier_removal:
                if save_all:
                    for i in range(1, all_label_likelihoods.shape[0]):
                        all_label_likelihoods[i, :, :] = self.remove_outliers(
                            all_label_likelihoods[i, :, :], G_gpu)
                else:
                    all_label_likelihoods[-1, :, :] = self.remove_outliers(
                        all_label_likelihoods[-1, :, :], G_gpu)


            return PointCloudSegmentationResult(points=point_cloud, projected=projected,
                                                label_likelihoods=all_label_likelihoods,
                                                class_ids=detections.class_ids,
                                                in_camera_view=in_view,
                                                initial_labels=initial_point_labels)
