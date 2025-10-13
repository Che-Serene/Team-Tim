#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from enum import Enum
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger
from sdk.common import plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import RGBState, RGBStates
import onnxruntime as ort
from colorzero import Color
from gpiozero import RGBLED

# 로봇의 현재 '단계(Phase)'를 정의
class RobotPhase(Enum):
    LANE_FOLLOWING = 0
    STOP_AND_CHECK = 1
    EXECUTING_TURN = 2
    PARKING_ACTION = 3
    EXECUTING_PASS = 4
    PARKING_STOP = 5

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        
        self.pid_straight = pid.PID(0.5, 0.01, 0.2)
        self.pid_corner = pid.PID(2.5, 0.0, 0.8)    
        self.lock = threading.Lock()
        self.param_init()

        self.fps = fps.FPS()  
        self.display = True
        self.bridge = CvBridge()
        
        self.latest_cv_image = None
        self.last_detections = []
        
        self.declare_parameter('publish_result_image', True)

        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 1)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        
        timer_cb_group = ReentrantCallbackGroup()
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)
        
        model_path = "/home/ubuntu/ros2_ws/src/yolov5_ros2/config/nano.onnx" 
        self.get_logger().info(f"Loading ONNX model from {model_path}")
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            self.ort_session = None

        self.model_input_size = (640, 640)
        self.classes = ['cross', 'go', 'green', 'off', 'parking', 'red', 'right', 'yellow']
        
        self.rgb_pub = self.create_publisher(RGBStates, '/ros_robot_controller/set_rgb', 10)
        self.update_led("")
        self.led_toggle = False
        self.blink_timer = None

    def init_process(self):
        self.timer.cancel()
        self.mecanum_pub.publish(Twist())
        time.sleep(1)
        
        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main_driver_loop, daemon=True).start()
        threading.Thread(target=self.object_detection_navigator_loop, daemon=True).start()
        threading.Thread(target=self.led_rainbow_loop, daemon=True).start()

        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    # [핵심 수정 1] '초록불 대기' 상태 변수 추가
    def param_init(self):
        self.start = False
        self.corner_detected = False
        self.go_detected = False
        self.parking_detected = False # right_detected 변수 추가
        self.normal_speed = 0.8
        self.action_start_time = 0.0
        self.right_turn_state = 0
        self.current_phase = RobotPhase.LANE_FOLLOWING
        self.waiting_for_green_light = True # True로 시작하여 초록불을 기다림
        self.gpio_led_state = 'red'  
        with self.lock:
            self.last_detections = []

    def get_node_state(self, request, response):
        response.success = True
        return response

    def enter_srv_callback(self, request, response):
        self.get_logger().info("self driving enter")
        self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        self.mecanum_pub.publish(Twist())
        response.success = True
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info("self driving exit")
        self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info("set_running")
        with self.lock:
            self.start = request.data
            if self.start:
                self.waiting_for_green_light = True # 서비스로 시작 시 항상 대기 상태로
        response.success = True
        return response

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        with self.lock:
            self.latest_cv_image = cv_image
    
    def lidar_callback(self, msg: LaserScan):
        start_angle_deg = -4.0 
        end_angle_deg = 1.0
        start_angle_rad = math.radians(start_angle_deg)
        end_angle_rad = math.radians(end_angle_deg)

        try:
            start_index = int((start_angle_rad - msg.angle_min) / msg.angle_increment)
            end_index = int((end_angle_rad - msg.angle_min) / msg.angle_increment)
            start_index = max(0, start_index)
            end_index = min(len(msg.ranges) - 1, end_index)
        except (ValueError, ZeroDivisionError): return
        if start_index >= end_index: return
        
        corner_in_range = False
        go_distance_detected = False
        parking_distance_detected = False

        for distance in msg.ranges[start_index:end_index]:
            self.distance = distance
            if 0 < distance < 0.9:
                corner_in_range = True
            if 0.8 < distance < 1.4:
                parking_distance_detected = True
            if 2.15 < distance < 2.4:
                go_distance_detected = True
        
        with self.lock:
            self.corner_detected = corner_in_range
            self.go_detected = go_distance_detected
            self.parking_detected = parking_distance_detected
            
    def led_rainbow_loop(self):
        leds = [
            RGBLED(2,3,4, active_high=False), RGBLED(17,27,22, active_high=False),
            RGBLED(10,9,11, active_high=False), RGBLED(0,5,6, active_high=False),
            RGBLED(13,19,26, active_high=False), RGBLED(14,15,18, active_high=False),
        ]
        self.get_logger().info("GPIO LED control thread started.")
        hue = 0
        try:
            while self.is_running and rclpy.ok():
                # self.gpio_led_state 값에 따라 LED 색상을 다르게 제어
                state = self.gpio_led_state
                
                if state == 'rainbow':
                    # 주행 중: 무지개 효과
                    hue = (hue + 5) % 360
                    for i, led in enumerate(leds):
                        hue_val = ((hue + i * 30) % 360) / 360.0
                        led.color = Color.from_hsv(hue_val, 1.0, 0.5)
                    time.sleep(0.01)
                
                elif state == 'yellow':
                    # 코너링 중: 노란색
                    for led in leds:
                        led.color = (1.0, 1.0, 0.0)
                    time.sleep(0.1)

                elif state == 'red':
                    # 정지 중: 빨간색
                    for led in leds:
                        led.color = (1.0, 0.0, 0.0)
                    time.sleep(0.1)

                elif state == 'off':
                    # 주차 완료: 꺼짐
                    for led in leds:
                        led.color = (0, 0, 0)
                    time.sleep(0.1)
        finally:
            for led in leds: led.off()

    def update_led(self, led: str = "off"):
        msg = RGBStates()
        match led:
            case "rd":
                color = (255, 0, 0)  # 빨강
            case "gr":
                color = (0, 255, 0)  # 초록
            case "yl":
                color = (255, 255, 0)  # 노랑
            case "gr_off":
                color = (0, 255, 0)  # 초록
            case "off":
                color = (0, 0, 0)  # 끔
            case _:
                color = (148, 0, 211)  # 보라: 뭔가 문제가 생김;
        
        if led == "yl":
            msg.states = [
                RGBState(index=1, red=0, green=0, blue=0),
                RGBState(index=2, red=color[0], green=color[1], blue=color[2])
            ]
            self.rgb_pub.publish(msg)
        elif led == "gr_off":
            msg.states = [
                RGBState(index=1, red=color[0], green=color[1], blue=color[2]),
                RGBState(index=2, red=0, green=0, blue=0)
            ]
            self.rgb_pub.publish(msg)
        else:
            msg.states = [
                RGBState(index=1, red=color[0], green=color[1], blue=color[2]),
                RGBState(index=2, red=color[0], green=color[1], blue=color[2])
            ]
            self.rgb_pub.publish(msg)

    def toggle_led_cb(self):
        if self.led_toggle:
            self.update_led("yl")
        else:
            self.update_led("gr_off")
        self.led_toggle = not self.led_toggle

    def preprocess(self, img):
        img_h, img_w, _ = img.shape
        scale = min(self.model_input_size[0] / img_w, self.model_input_size[1] / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        padded_img = np.full((self.model_input_size[1], self.model_input_size[0], 3), 114, dtype=np.uint8)
        padded_img[(self.model_input_size[1] - new_h) // 2:(self.model_input_size[1] - new_h) // 2 + new_h,
                   (self.model_input_size[0] - new_w) // 2:(self.model_input_size[0] - new_w) // 2 + new_w, :] = resized_img
        image_data = np.array(padded_img, dtype='float32') / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)
        return image_data, scale, (new_w, new_h)

    def postprocess(self, output, original_shape, scale, new_shape):
        img_h, img_w = original_shape
        new_w, new_h = new_shape
        pad_x = (self.model_input_size[0] - new_w) / 2
        pad_y = (self.model_input_size[1] - new_h) / 2
        boxes, scores, class_ids = [], [], []
        
        for i in range(output.shape[1]):
            prediction = output[0, i]
            confidence = prediction[4]
            if confidence > 0.25: # 신뢰도 낮춰서 테스트
                class_score = np.max(prediction[5:])
                class_id = np.argmax(prediction[5:])
                if class_score > 0.1: # 신뢰도 낮춰서 테스트
                    cx, cy, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
                    x1 = int(((cx - w / 2) - pad_x) / scale)
                    y1 = int(((cy - h / 2) - pad_y) / scale)
                    width = int(w / scale)
                    height = int(h / scale)
                    boxes.append([x1, y1, width, height])
                    scores.append(float(confidence * class_score))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.5)
        detections = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            detections.append({'class_name': self.classes[class_ids[i]], 'score': scores[i], 'box': [x, y, x + w, y + h]})
        return detections

    # [핵심 수정 2] 'green' 신호를 감지하여 대기 상태 해제
    def object_detection_navigator_loop(self):
        while self.is_running and rclpy.ok():
            image = None
            with self.lock:
                if self.latest_cv_image is not None and self.start:
                    image = self.latest_cv_image.copy()

            if image is not None and self.ort_session:
                input_tensor, scale, new_shape = self.preprocess(image)
                outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
                detections = self.postprocess(outputs[0], image.shape[:2], scale, new_shape)
                
                with self.lock:
                    self.last_detections = detections
                    
                    if self.waiting_for_green_light:
                        visible_signs = {det['class_name'] for det in detections}
                        if 'green' in visible_signs:
                            self.get_logger().warn("<<<<< GREEN LIGHT DETECTED! STARTING RUN! >>>>>")
                            self.waiting_for_green_light = False

    def main_driver_loop(self):
        while self.is_running and rclpy.ok():
            phase = None
            local_is_corner = False
            local_go_detected = False
            local_parking_detected = False
            image = None
            is_waiting = True
            
            with self.lock:
                if self.latest_cv_image is not None and self.start:
                    image = self.latest_cv_image.copy()
                is_waiting = self.waiting_for_green_light
                phase = self.current_phase
                local_is_corner = self.corner_detected
                local_go_detected = self.go_detected
                local_parking_detected = self.parking_detected

            if image is None or not self.start:
                time.sleep(0.01)
                continue
            
            # [핵심 수정 3] 대기 상태일 경우, 주행 로직을 건너뛰고 정지
            if is_waiting:
                self.get_logger().info("--- WAITING FOR GREEN LIGHT TO START ---", throttle_duration_sec=1)
                self.update_led("rd") # 빨강
                self.gpio_led_state = 'red'
                self.mecanum_pub.publish(Twist()) # 정지
                # 화면 표시는 계속하도록 아래로 넘어감
            else:
                # --- 주행 로직 ---
                result_image = image.copy()
                binary_image = self.lane_detect.get_binary(image)
                result_image, lane_angle, lane_x, lane_y = self.lane_detect(binary_image, result_image)
                
                twist = Twist()

                if phase == RobotPhase.LANE_FOLLOWING:

                    if local_go_detected:
                        # 교차로 진입 시에는 깜빡이를 끕니다.
                        if self.blink_timer is not None:
                            self.blink_timer.cancel()
                            self.blink_timer = None
                        with self.lock:
                            self.get_logger().warn(f"Intersection distance met. ==> Phase: STOP_AND_CHECK")
                            self.current_phase = RobotPhase.STOP_AND_CHECK
                            self.action_start_time = time.time()
                    else:
                        if lane_x >= 0:
                            if local_is_corner:
                                self.gpio_led_state = 'yellow' 
                                # 코너 상태이고, 깜빡이가 아직 안 켜져 있으면 새로 생성합니다.
                                if self.blink_timer is None:
                                    self.get_logger().info("Corner detected, starting blink timer.")
                                    self.blink_timer = self.create_timer(0.15, self.toggle_led_cb)
                                    self.led_toggle = True # 바로 켜지도록 True로 시작
                                
                                self.get_logger().info("--- Phase: LANE_FOLLOWING (Cornering) ---")
                                self.pid_corner.SetPoint = 150
                                self.pid_corner.update(lane_x)
                                twist.linear.x = self.normal_speed * 0.8
                                twist.angular.z = common.set_range(self.pid_corner.output, -2.45, 2.45)
                                
                            else: # 직선 주로 상태
                                self.gpio_led_state = 'rainbow'
                                # 직선 주로이고, 깜빡이가 켜져 있으면 끕니다.
                                if self.blink_timer is not None:
                                    self.get_logger().info("Straight lane, canceling blink timer.")
                                    self.blink_timer.cancel()
                                    self.blink_timer = None
                                
                                # 깜빡이가 꺼진 후에는 초록불을 켭니다.
                                self.update_led("gr")
                                self.get_logger().info("--- Phase: LANE_FOLLOWING (Straight) ---")
                                self.pid_straight.SetPoint = 150
                                self.pid_straight.update(lane_x)
                                twist.linear.x = self.normal_speed
                                twist.angular.z = common.set_range(self.pid_straight.output, -0.25, 0.25)
                        else:
                            self.pid_straight.clear()
                            self.pid_corner.clear()
                elif phase == RobotPhase.STOP_AND_CHECK:
                    self.update_led("rd") # 빨강
                    self.gpio_led_state = 'red'
                    self.get_logger().info(f"--- Phase: STOP_AND_CHECK ---")
                    twist.linear.x = 0.0

                    elapsed_time = time.time() - self.action_start_time
                    if elapsed_time > 1.2:
                        visible_signs = {det['class_name'] for det in self.last_detections}
                        self.get_logger().info(f"Checking signs. Seeing: {visible_signs}")
                        

                        if 'green' in visible_signs:
                            with self.lock:
                                self.get_logger().warn(f"Green sign detected. ==> Phase: EXECUTING_PASS")
                                self.current_phase = RobotPhase.EXECUTING_PASS
                                self.action_start_time = time.time()

                        elif 'right' in visible_signs:
                            with self.lock:
                                self.get_logger().warn(f"Right sign detected. ==> Phase: EXECUTING_TURN")
                                self.current_phase = RobotPhase.EXECUTING_TURN
                                self.action_start_time = time.time()
                                self.right_turn_state = 1
                        elif 'go' in visible_signs:
                            with self.lock:
                                self.get_logger().warn(f"Right sign detected. ==> Phase: EXECUTING_PASS")
                                self.current_phase = RobotPhase.EXECUTING_PASS
                                self.action_start_time = time.time()

                        elif len({'red', 'off', 'yellow'} & visible_signs) >= 1 :

                            if 2.0>self.distance >= 1.8:
                                twist.linear.x = 0.0
                                time.sleep(0.1)
                        


                        else:
                            with self.lock:
                                self.get_logger().info(f"Default action (Go or No sign). ==> Phase: EXECUTING_PASS")
                                self.current_phase = RobotPhase.EXECUTING_PASS
                                self.action_start_time = time.time()
                
                elif phase == RobotPhase.EXECUTING_TURN:
                    self.get_logger().info(f"--- Phase: EXECUTING_TURN (Step: {self.right_turn_state}) ---")
                    elapsed_time = time.time() - self.action_start_time
                    step1_end, step2_end = 1.9, 4.21
                    self.gpio_led_state = 'yellow'
                    if self.blink_timer is None:
                        self.blink_timer = self.create_timer(0.15, self.toggle_led_cb)
                        self.led_toggle = False

                    if self.right_turn_state == 1:
                        acceleration_duration = 1.0
                        if elapsed_time < acceleration_duration:
                            speed_ratio = elapsed_time / acceleration_duration
                            twist.linear.x = (self.normal_speed * 0.6) * speed_ratio
                        else:
                            twist.linear.x = self.normal_speed * 0.6
                        if elapsed_time > step1_end: self.right_turn_state = 2

                    elif self.right_turn_state == 2:
                        twist.linear.x = 0.1
                        twist.angular.z = -0.7
                        if elapsed_time > step2_end:
                            self.right_turn_state = 3
                    
                    if self.right_turn_state == 3:
                        self.gpio_led_state = 'rainbow'
                        if self.blink_timer is not None:
                            self.blink_timer.cancel()
                            self.blink_timer = None
                        twist.linear.x = self.normal_speed
                            
                        if lane_x >= 0:
                            # self.pid_straight.SetPoint = 145
                            # self.pid_straight.update(lane_x)
                            # twist.angular.z = common.set_range(self.pid_straight.output, -0.2, 0.2)
                            
                            self.get_logger().info("--- Phase: LANE_FOLLOWING (Straight-parking) ---")
                            self.pid_straight.SetPoint = 150
                            self.pid_straight.update(lane_x)
                            twist.angular.z = common.set_range(self.pid_straight.output, -0.25, 0.25)
                        if local_parking_detected:
                            twist = Twist()
                            twist.linear.y = -0.2
                            self.mecanum_pub.publish(twist)
                            time.sleep(2.3)
                            self.current_phase = RobotPhase.PARKING_STOP

                elif phase == RobotPhase.PARKING_STOP:
                    self.gpio_led_state = 'off' 
                    self.update_led("off") # 끔
                    twist = Twist()
                    self.mecanum_pub.publish(twist)

                elif phase == RobotPhase.EXECUTING_PASS:
                    self.gpio_led_state = 'rainbow'
                    self.update_led("gr") # 초록
                    self.get_logger().info(f"--- Phase: EXECUTING_PASS ---")
                    elapsed_time = time.time() - self.action_start_time
                    
                    acceleration_duration = 1.0
                    if elapsed_time < 1.0:
                        twist.linear.x = self.normal_speed
                            
                        if lane_x >= 0:
                            self.pid_straight.SetPoint = 145
                            self.pid_straight.update(lane_x)
                            twist.angular.z = common.set_range(self.pid_straight.output, -0.2, 0.2)
                    else:
                        with self.lock:
                            self.get_logger().warn("Pass finished. ==> Phase: LANE_FOLLOWING")
                            self.current_phase = RobotPhase.LANE_FOLLOWING
                    

                self.mecanum_pub.publish(twist)

            # --- 화면 업데이트는 대기 중이든 주행 중이든 항상 실행 ---
            result_image = image.copy()
            with self.lock:
                detections_to_draw = self.last_detections
            if detections_to_draw:
                for det in detections_to_draw:
                    plot_one_box(det['box'], result_image, label=f"{det['class_name']}:{det['score']:.2f}")

            if self.get_parameter('publish_result_image').value:
                bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)
                self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
        if self.blink_timer is not None:
            self.blink_timer.cancel()
        self.mecanum_pub.publish(Twist())

def main():
    node = None
    try:
        node = SelfDrivingNode('self_driving')
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('KeyboardInterrupt, shutting down...')
    finally:
        if node:
            node.is_running = False
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
if __name__ == "__main__":
    main()
