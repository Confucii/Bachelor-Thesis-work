import torch
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()

length_joint = 2.0
target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])
segment = np.array([0, 1]) * length_joint
change = 0.01

is_running = True


def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])


def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app


fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

theta_1 = np.deg2rad(0)
theta_2 = np.deg2rad(0)
theta_3 = np.deg2rad(0)


def rotation(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    R = np.array([
        [cos, -sin],
        [sin, cos]
    ])
    return R


'''def d_rotation(theta):
    change = 0.01
    dR = (rotation(theta + change) - rotation(theta)) / change
    return dR'''


def d_rotation(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    dR = np.array([
        [-sin, -cos],
        [cos, -sin]
    ])
    return dR

while is_running:
    plt.clf()

    joint1 = np.dot(rotation(theta_1), segment)
    joint2 = np.dot(rotation(theta_1), (segment + np.dot(rotation(theta_2), segment)))
    joint3 = np.dot(rotation(theta_1), (segment + np.dot(rotation(theta_2), (segment + np.dot(rotation(theta_3), segment)))))

    joint1_d_rot = np.dot(d_rotation(theta_1), segment)
    joint2_d_rot = np.dot(d_rotation(theta_2), segment)
    joint3_d_rot = np.dot(d_rotation(theta_3), segment)

    deriv = np.sum(-2 * (target_point - joint3) * joint1_d_rot) # speed : (-2) , target_point - joint3 : (distance), joint1_d_rot : (direction)
    deriv2 = np.sum(-2 * (target_point - joint3) * np.dot(rotation(theta_1), joint2_d_rot))
    deriv3 = np.sum(-2 * (target_point - joint3) * np.dot(rotation(theta_1), np.dot(rotation(theta_2), joint3_d_rot)))
    loss = np.sum((target_point - joint3) ** 2)

    theta_1 = theta_1 - change * deriv
    theta_2 = theta_2 - change * deriv2
    theta_3 = theta_3 - change * deriv3

    joints = []
    joints.append(anchor_point)
    joints.append(joint1)
    joints.append(joint2)
    joints.append(joint3)
    np_joints = np.array(joints)


    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.title(f'loss: {loss} theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))} theta_3: {round(np.rad2deg(theta_3))}')
    plt.draw()
    plt.pause(1e-3)

    #break

