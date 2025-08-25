import pygame

import lcm
from lcm_t.exlcm import twist_t
import time

import numpy as np
import threading

lc = lcm.LCM()

vel_global = np.array((0.0, 0.0, 0.0)) 
lock = threading.Lock()

def send_data():
    interval = 1 / 22 # 22hz
    x_v = 0
    y_v = 0
    ang_v = 0
    while True:
        with lock:  # Ensure thread-safe read
            x_v = vel_global[0]
            y_v = vel_global[1]
            ang_v = vel_global[2]
        
        # Send lcm twist command
        msg = twist_t()
        msg.x_vel[0] = x_v
        msg.y_vel[0] = y_v
        msg.omega_vel[0] = ang_v
        lc.publish("TWIST_T", msg.encode())
        print('lcm sent\n')

        time.sleep(interval)

thread = threading.Thread(target=send_data, daemon=True)
thread.start()

pygame.init()

# Initialize the joystick module
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")

# Screen settings
width, height = 400, 250
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("joystick ctr")

white = (255, 255, 255)
black = (0, 0, 0)
orange = (255, 165, 0)

bar_width = 50
bar_spacing = 20
base_y = height - 10

# Velocity magnitude
linear_vv = 1.5
angular_vv = 1.5

clock = pygame.time.Clock()

# Main loop
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Exit condition
            running = False

    # Read joystick axes (left stick, right stick, triggers)
    left_stick_x = joystick.get_axis(0)  # Left stick X-axis
    left_stick_y = joystick.get_axis(1)  # Left stick Y-axis
    right_stick_x = joystick.get_axis(3)  # Right stick X-axis
    right_stick_y = joystick.get_axis(4)  # Right stick Y-axis
    left_trigger = joystick.get_axis(2)  # Left trigger
    right_trigger = joystick.get_axis(5)  # Right trigger
    # Read button states
    button_a = joystick.get_button(0)  # A button
    button_b = joystick.get_button(1)  # B button
    button_x = joystick.get_button(2)  # X button
    button_y = joystick.get_button(3)  # Y button
    # Read D-pad
    dpad = joystick.get_hat(0)  # D-pad direction

    # rumble the controller
    strength_ctr_all = (abs(left_stick_x)+abs(left_stick_y)+abs(right_stick_x))/3
    #joystick.rumble(0.0, 0.5*strength_ctr_all, 0) # low freq motor, high freq motor, time

    y_velocity = -1*left_stick_y*linear_vv
    x_velocity = -1*left_stick_x*linear_vv
    angular_velocity = -1*right_stick_x*angular_vv

    with lock:  # Ensure thread-safe read
        vel_global[0] = y_velocity
        vel_global[1] = x_velocity
        vel_global[2] = angular_velocity

    # Clear the screen
    screen.fill(white)

    # Calculate bar heights
    x_bar_height = int(abs(left_stick_y) * 200)  
    y_bar_height = int(abs(left_stick_x) * 200)
    angular_bar_height = int(abs(right_stick_x) * 200)

    # Draw the bars
    pygame.draw.rect(screen, black, (50, base_y - x_bar_height, bar_width, x_bar_height))
    pygame.draw.rect(screen, black, (150, base_y - y_bar_height, bar_width, y_bar_height))
    pygame.draw.rect(screen, black, (250, base_y - angular_bar_height, bar_width, angular_bar_height))

    # Add labels
    font = pygame.font.Font(None, 24)
    x_label = font.render("X Vel", True, black)
    y_label = font.render("Y Vel", True, black)
    angular_label = font.render("Ang Vel", True, black)

    screen.blit(x_label, (50, 10))
    screen.blit(y_label, (150, 10))
    screen.blit(angular_label, (250, 10))

    # Update the display
    pygame.display.flip()

    print(f"X v: {x_velocity:.2f}, Y v: {y_velocity:.2f}, Angular v: {angular_velocity:.2f}")

    clock.tick(30)

joystick.rumble(0, 0, 0)
pygame.quit()
