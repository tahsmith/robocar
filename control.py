from shutil import rmtree

import Adafruit_PCA9685
import time

from math import sin, pi

import os
import picamera


def map_range(x, X_min, X_max, Y_min, Y_max):
    '''
    Linear mapping between two ranges of values
    '''
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range / Y_range

    y = ((x - X_min) / XY_ratio + Y_min) // 1

    return int(y)


class PCA9685:
    '''
    PWM motor controller using PCA9685 boards.
    This is used for most RC Cars
    '''

    def __init__(self, channel, frequency=60):
        # Initialise the PCA9685 using the default address (0x40).
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel

    def set_pulse(self, pulse):
        self.pwm.set_pwm(self.channel, 0, pulse)

    def run(self, pulse):
        self.set_pulse(pulse)


class PWMSteering:
    """
    Wrapper over a PWM motor contoller to convert angles to PWM pulses.
    """
    LEFT_ANGLE = -1
    RIGHT_ANGLE = 1

    def __init__(self, controller=None,
                 left_pulse=290,
                 right_pulse=490):
        self.controller = controller
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse

    def run(self, angle):
        # map absolute angle to angle that vehicle can implement.
        pulse = map_range(angle,
                          self.LEFT_ANGLE, self.RIGHT_ANGLE,
                          self.left_pulse, self.right_pulse)

        self.controller.set_pulse(pulse)

    def shutdown(self):
        self.run(0)  # set steering straight


class PWMThrottle:
    """
    Wrapper over a PWM motor controller to convert -1 to 1 throttle
    values to PWM pulses.
    """
    MIN_THROTTLE = -1
    MAX_THROTTLE = 1

    def __init__(self, controller=None,
                 max_pulse=300,
                 min_pulse=490,
                 zero_pulse=350):

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse

        # send zero pulse to calibrate ESC
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)

    def run(self, throttle):
        if throttle > 0:
            pulse = map_range(throttle,
                              0, self.MAX_THROTTLE,
                              self.zero_pulse, self.max_pulse)
        else:
            pulse = map_range(throttle,
                              self.MIN_THROTTLE, 0,
                              self.min_pulse, self.zero_pulse)

        self.controller.set_pulse(pulse)

    def shutdown(self):
        self.run(0)  # stop vehicle


import numpy as np
from image import *


def calculate_steering(image):
    left, right = get_lane_lines(image)
    if left is None and right is None:
        return None

    if left is None:
        return -1
    if right is None:
        return 1

    return (x_intercept(right) + x_intercept(left) - image_size[1] / 2) / image_size[1]


def calculate_steering_2(image):
    left, right = split_lines(hough_lines(image))
    if left.shape[0] == 0 and right.shape[0] == 0:
        return None

    if left.shape[0] == 0:
        return -1
    if right.shape[0] == 0:
        return 1

    return (np.mean(left[:, 0]) + np.mean(left[:, 2]) + np.mean(right[:, 2]) - image_size[1] / 2) / image_size[1]

k = 1.0
throttle_scale = -0.2
steering_scale = -1.0

if os.path.isdir('./output'):
    rmtree('./output')


os.mkdir('./output')


def control_loop(camera, steering_controller, throttle_controller):
    image = np.empty(image_size + (3,), dtype=np.uint8)
    steering = 0
    i = 0
    while True:
        for _ in camera.capture_continuous(image, 'rgb', use_video_port=True):
            steering_new = calculate_steering_2(image)
            if steering_new is None:
                steering_new = steering

            steering = k * steering_new + (1 - k) * steering
            steering_controller.run(steering_scale * steering)
            # throttle = 1 - 0.5 * steering ** 2
            throttle = 1
            throttle_controller.run(throttle_scale * throttle)

            print(steering, throttle)
            i += 1
            cv2.imwrite('./output/{0}.jpg'.format(i), image)


def main():
    steering_controller = PWMSteering(PCA9685(1))
    throttle_controller = PWMThrottle(PCA9685(0))

    with picamera.PiCamera() as camera:
        camera.resolution = image_size[::-1]
        camera.framerate = 24
        try:
            control_loop(camera, steering_controller, throttle_controller)
        finally:
            steering_controller.shutdown()
            throttle_controller.shutdown()


if __name__ == '__main__':
    main()
