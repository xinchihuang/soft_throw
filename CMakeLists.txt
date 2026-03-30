cmake_minimum_required(VERSION 3.0.2)
project(soft_throw)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
  control_msgs
  trajectory_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  real/run_min_throw_real.py
  scripts/send_trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
