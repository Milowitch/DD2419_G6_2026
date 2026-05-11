#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from robp_interfaces.msg import ArmControl
from sensor_msgs.msg import JointState

import sys
import termios
import tty
import select
import time

class Arm_Control(Node):
    def __init__(self):
        super().__init__('arm_control')
        self.publisher = self.create_publisher(
            ArmControl,
            '/arm/control',   
            10
        )

    def move_arm(self, positions, time_ms=3000):
        """
        positions: list of 6 joint angles (degrees)
        time_ms: int or list of 6 ints (milliseconds)
        """

        if len(positions) != 6:
            self.get_logger().error("Expected 6 joint positions")
            return

        msg = ArmControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = positions

        # Time
        if isinstance(time_ms, int):
            msg.time = [time_ms] * 6
        else:
            msg.time = time_ms

        self.publisher.publish(msg)
        self.get_logger().info(f"Sent arm command: {positions}")

    def pick_up_object(self):
        self.move_arm([40, 120, 30, 220, 180, 120], time_ms=3000)
        time.sleep(10)
        self.move_arm([40, 120, 50, 170, 60, 120], time_ms= [3000, 3000, 3000, 3000, 7000, 3000])
        time.sleep(10)
        self.move_arm([110, 120, 50, 170, 60, 120], time_ms=3000)

        # if self.gripper_closed():
        #     self.get_logger().info("Object picked up")
        # else:
        #     self.get_logger().warn("Pickup failed")

        time.sleep(4)
        self.move_arm([110, 120, 30, 220, 180, 120], time_ms=3000)
        time.sleep(4)
        self.move_arm([110, 120, 30, 220, 180, 120], time_ms=3000)


    def arm_reset(self):
        self.move_arm([40, 120, 30, 220, 180, 120], time_ms=3000)

    # def gripper_closed(self):
    # # closed if angle < threshold
    #     return True if self.last_positions[5] < 109 else False

        

def main():
    rclpy.init()
    node = Arm_Control()
    node.pick_up_object()  

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

