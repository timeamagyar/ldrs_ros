std_msgs/Header header

# Integer instance IDs for each segmented object
# Identifies corresponding Mask-RCNN instance for each object
int32[] instance_ids

# Integer class IDs for each segmented object
int32[] class_ids

# String class IDs for each segmented object
string[] class_names

# Bools indicating wether individual Lidar points are in the view of the camera
bool[] in_camera_view

# 3D segmentations as PointCloud2
sensor_msgs/PointCloud2[] object_points

# 3D points in camera view
sensor_msgs/PointCloud2 points

