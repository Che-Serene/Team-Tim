#!/usr/bin/env python3
import os
import cv2
import math
import time
import queue
import rclpy
import numpy as np
import sdk.pid as pid
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from example.self_driving import lane_detect


class Boongboong2(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 핵심 제어 변수들
        self.is_running = True
        self.is_enabled = True
        
        # PID 제어기 설정
        self.pid_go = pid.PID(0.6, 0.0, 0.15)
        self.pid_go.SetPoint = 130
        
        # 영상 처리
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # 탐지된 객체
        self.crosswalk = False
        self.traffic_light = None
        self.traffic_sign = None

        # 제어 상태
        self.stop = False
        self.normal_speed = 0.8
        self.slow_down_speed = 0.1
        
        # 쿨다운 관리
        self.last_crosswalk_time = 0
        self.crosswalk_cooldown = 5.0

        # Publisher
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        
        # Subscribers
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.object_callback, 1)
        
        self.get_logger().info('🚗 Boongboong2 initialized!')

    def image_callback(self, ros_image):
        """카메라 이미지 수신"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            rgb_image = np.array(cv_image, dtype=np.uint8)
            
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(rgb_image)
        except Exception as e:
            self.get_logger().error(f'Image error: {e}')

    def object_callback(self, msg):
        """YOLO 객체 감지 결과 수신"""
        self.objects_info = msg.objects
        
        # 매번 초기화
        self.crosswalk = False
        self.traffic_light = None
        self.traffic_sign = None
        
        if self.objects_info:
            crosswalks = [obj for obj in self.objects_info if obj.class_name == 'cross']
            traffic_lights = [obj for obj in self.objects_info if obj.class_name in ('green', 'off', 'red')]
            traffic_signs = [obj for obj in self.objects_info if obj.class_name in ('go', 'right', 'parking')]
            
            if len(crosswalks) >= 2:
                largest_crosswalk = max(crosswalks, key=lambda obj: abs(obj.box[2] - obj.box[0]))
                center_y = int((largest_crosswalk.box[1] + largest_crosswalk.box[3]) / 2)
                if 390 <= center_y <= 450:
                    if time.time() - self.last_crosswalk_time > self.crosswalk_cooldown:
                        self.crosswalk = True

            if traffic_signs:
                largest_sign = max(traffic_signs, key=lambda obj: abs(obj.box[3] - obj.box[1]))
                if abs(largest_sign.box[3] - largest_sign.box[1]) > 50:
                    self.traffic_sign = largest_sign.class_name

            if traffic_lights:
                largest_light = max(traffic_lights, key=lambda obj: abs(obj.box[2] - obj.box[0]))
                self.traffic_light = largest_light.class_name

    def should_turn(self, lane_x):
        return (self.traffic_sign == 'right' or (lane_x >= 0 and lane_x > 150))

    def park_action(self):
        twist = Twist()
        twist.linear.y = -0.2
        return twist

    def turn_action(self):
        twist = Twist()
        twist.linear.x = self.slow_down_speed
        twist.angular.z = -0.45
        return twist

    def go_straight(self, lane_x):
        twist = Twist()
        twist.linear.x = self.normal_speed
        
        if lane_x >= 0:
            self.pid_go.update(lane_x)
            twist.angular.z = common.set_range(self.pid_go.output, -0.15, 0.15)
        else:
            self.pid_go.clear()
            
        return twist


def main():
    """🎯 main에서 직접 반복문 실행"""
    rclpy.init()  # ✅ 여기서 초기화
    
    node = Boongboong2('boongboong2')
    
    try:
        while rclpy.ok():
            start_time = time.time()
            
            # ✅ ROS2 콜백 처리 (논블로킹)
            rclpy.spin_once(node, timeout_sec=0.01)
            
            try:
                # 이미지 처리
                frame = node.image_queue.get(timeout=0.01)
                
                # 차선 감지
                binary_image = node.lane_detect.get_binary(frame)
                result_image, lane_angle, lane_x = node.lane_detect(binary_image, frame.copy())
                
            except queue.Empty:
                lane_x = -1
                time.sleep(0.01)
                continue
            except Exception as e:
                node.get_logger().error(f'Image processing error: {e}')
                time.sleep(0.01)
                continue

            # 🎯 정지 조건 우선 처리
            current_time = time.time()
            
            # 횡단보도: 1초 무조건 멈춤
            if node.crosswalk and not hasattr(node, 'crosswalk_stop_start'):
                node.crosswalk_stop_start = current_time
                node.last_crosswalk_time = current_time
                node.get_logger().info('🚶 Crosswalk - 1sec stop')
            
            if hasattr(node, 'crosswalk_stop_start'):
                if current_time - node.crosswalk_stop_start < 1.0:
                    node.mecanum_pub.publish(Twist())
                    time.sleep(1/30)
                    continue
                else:
                    delattr(node, 'crosswalk_stop_start')
                    node.get_logger().info('✅ Crosswalk stop completed')
            
            # 빨간불: 초록불까지 대기
            if node.traffic_light == 'red':
                node.mecanum_pub.publish(Twist())
                node.get_logger().info('🚦 Red light - STOP')
                time.sleep(1/30)
                continue
            
            # 🎯 일반 주행 로직
            try:
                if node.traffic_sign == 'parking':
                    twist = node.park_action()
                    node.get_logger().info('🅿️ Parking')
                    
                elif node.should_turn(lane_x):
                    if not hasattr(node, 'turn_start_time'):
                        node.turn_start_time = current_time
                        node.get_logger().info(f'🔄 Starting turn - lane_x: {lane_x}')
                    
                    twist = node.turn_action()
                    
                    # 3초 후 턴 완료
                    if current_time - node.turn_start_time > 3.0:
                        delattr(node, 'turn_start_time')
                        node.traffic_sign = None
                        node.get_logger().info('✅ Turn completed')
                    
                else:
                    twist = node.go_straight(lane_x)
                
                # 명령 발행
                if node.is_enabled:
                    node.mecanum_pub.publish(twist)
                    
            except Exception as e:
                node.get_logger().error(f'Control logic error: {e}')
            
            # 30Hz 제어
            processing_time = time.time() - start_time
            sleep_time = 1/30 - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
