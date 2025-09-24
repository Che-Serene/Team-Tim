import sys
if sys.prefix == '/home/intel/sesac/d021rs-yolov5-ros2-Che-Serene/.venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/intel/sesac/d021rs-yolov5-ros2-Che-Serene/ros2_ws/install/peripherals'
