#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from robp_interfaces.msg import Encoders, DutyCycles
import numpy as np
import threading
import time

class DiffDrivePIFF(Node):
    def __init__(self):
        super().__init__('diffdrive_piff')


        self.r = 0.049      # wheel radius
        self.L = 0.3        # base width
        self.dt = 0.05

        self.ticks_per_rev = 48 * 64

        self.P_left = 0.05
        self.I_left = 0.1#0.1
        self.P_right = 0.05
        self.I_right = 0.1#0.1

        self.Kaw = 1.0   # anti-windup gain

        self.integral_left = 0.0
        self.integral_right = 0.0

        self.cmd_table = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

        self.speed_table_left = np.array([0.0,2.0,4.42,6.87,9.29,11.74,14.15,16.52,18.94,21.43,23.93])
        self.speed_table_right = np.array([0.0,2.21,4.70,7.28,9.77,12.31,14.85,17.34,19.84,22.58,25.03])


        self.delta_ticks_left = 0.0
        self.delta_ticks_right = 0.0

        self.v_ref = 0.0
        self.w_ref = 0.0

        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(Encoders, '/phidgets/motor/encoders', self.encoder_callback, 10)

        self.pub = self.create_publisher(DutyCycles, '/phidgets/motor/duty_cycles', 10)

        # Control thread
        threading.Thread(target=self.control_loop, daemon=True).start()

    def cmd_callback(self, msg):
        self.v_ref = msg.linear.x
        self.w_ref = msg.angular.z

    def encoder_callback(self, msg):
        self.delta_ticks_left = msg.delta_encoder_left
        self.delta_ticks_right = msg.delta_encoder_right

    def ticks_to_speed(self, delta_ticks):
        revs = delta_ticks / self.ticks_per_rev
        rad_s = revs * 2 * np.pi / self.dt
        return rad_s * self.r

    def ff_from_speed(self, speed_rad_s, table):
        sign = np.sign(speed_rad_s)
        return sign * np.interp(abs(speed_rad_s), table, self.cmd_table)

    def pi_aw(self, error, integral, P, I, ff):
        u_unsat = ff + P * error + I * integral
        u_sat = max(-1.0, min(1.0, u_unsat))

        # Anti-windup (back-calculation)
        integral += (error + self.Kaw * (u_sat - u_unsat)) * self.dt

        # Optional safety clamp
        integral = max(-2.0, min(2.0, integral))

        return u_sat, integral

    def control_loop(self):
        while rclpy.ok():

            vL_ref = self.v_ref - (self.L / 2.0) * self.w_ref
            vR_ref = self.v_ref + (self.L / 2.0) * self.w_ref

            vL = self.ticks_to_speed(self.delta_ticks_left)
            vR = self.ticks_to_speed(self.delta_ticks_right)


            eL = vL_ref - vL
            eR = vR_ref - vR

            # Deadband (reduces jitter)
            if abs(eL) < 0.02: eL = 0.0
            if abs(eR) < 0.02: eR = 0.0


            ffL = self.ff_from_speed(vL_ref / self.r, self.speed_table_left)
            ffR = self.ff_from_speed(vR_ref / self.r, self.speed_table_right)


            uL, self.integral_left = self.pi_aw(eL, self.integral_left, self.P_left, self.I_left, ffL)
            uR, self.integral_right = self.pi_aw(eR, self.integral_right, self.P_right, self.I_right, ffR)
            if abs(vL_ref) < 0.01: 
                uR = 0.0 
                self.integral_left =0
            if abs(vR_ref) < 0.01: 
                uL = 0.0 
                self.integral_right =0
            msg = DutyCycles()
            msg.duty_cycle_left = uL
            msg.duty_cycle_right = uR
            self.pub.publish(msg)

            time.sleep(self.dt)

def main():
    rclpy.init()
    node = DiffDrivePIFF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
