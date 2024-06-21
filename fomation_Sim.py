import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
dt = 0.1  # time step
L = 0.6  # distance between robots
RADIUS = 0.2  # radius of the robot
vmax = 0.2

# PID Controller class
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# Robot class
class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, vx, vy):
        self.x += vx * dt
        self.y += vy * dt
        self.theta = np.arctan2(vy, vx)  # Update theta based on the velocity direction

    def update_odom(self, vr, vl, w):
        dr = vr * dt
        dl = vl * dt
        dc = 0.5 * (dr + dl)
        self.theta += (dr - dl) / 0.2  # Increment theta by the change
        # self.theta = self.theta % (2 * np.pi)  # Normalize theta to be within 0 to 2π
        self.x += dc * np.cos(self.theta)
        self.y += dc * np.sin(self.theta)

# Initialize robots
leader = Robot(0, 0, 0)
follower = Robot(-L, 0, np.deg2rad(0))

# Initialize PID controllers
pid_vx = PID(10, 1.5, 0.1)
pid_vy = PID(10, 1.5, 0.1)
pid_w  = PID(15,1.5,0.1)
# Simulation
fig, ax = plt.subplots()
leader_circle = plt.Circle((leader.x, leader.y), RADIUS, fill=False, edgecolor='b')
follower_circle = plt.Circle((follower.x, follower.y), RADIUS, fill=False, edgecolor='r')
leader_arrow = ax.arrow(leader.x, leader.y, RADIUS * np.cos(leader.theta), RADIUS * np.sin(leader.theta), head_width=0.05, head_length=0.1, fc='b', ec='b')
follower_arrow = ax.arrow(follower.x, follower.y, RADIUS * np.cos(follower.theta), RADIUS * np.sin(follower.theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

ax.add_patch(leader_circle)
ax.add_patch(follower_circle)
def limit_velocity(v_r,v_l):
    if v_r > vmax: 
        v_r = vmax
    elif v_r < -vmax:
        v_r = -vmax

    if v_l > vmax : 
        v_l = vmax
    elif v_l < -vmax:
        v_l = -vmax
    return v_r,v_l
def init():
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    ax.grid(True)
    return leader_circle, follower_circle, leader_arrow, follower_arrow

def update(frame):
    global leader, follower, leader_circle, follower_circle, leader_arrow, follower_arrow

    # Leader's control (move in the x direction with constant speed)
    leader_vx = 0.14  # constant speed in x direction
    leader_vy = 0.14  # no movement in y direction
    # leader.w  = 0 
    # v_leader = vx *np.cos(leader.theta) + vy.np.sin(leader.theta)

    # vr_leader = v_leader 
    leader.update(leader_vx, leader_vy)
    # leader.update_odom()

    # Calculate the distance and angle error for the follower
    x_d = leader.x - L*np.cos(leader.theta)
    y_d = leader.y - L*np.sin(leader.theta)

    x_error = x_d - follower.x
    y_error = y_d - follower.y
    w_error = leader.theta - follower.theta

    vx = pid_vx.update(x_error)
    vy = pid_vy.update(y_error)
    # Calculate v and w for the follower
    v = vx*np.cos(follower.theta) + vy*np.sin(follower.theta)
    # theta_target = leader.theta
    # w = (theta_target - follower.theta) / dt
    w = (vx*np.cos(np.pi*1.5 - follower.theta) - vy*np.sin(np.pi*1.5 - follower.theta))/0.0325

    # w = pid_w.update(w_error)

    v_r = v + (w * 0.2/ 2)
    v_l = v - (w * 0.2 / 2)
    v_r, v_l = limit_velocity(v_r, v_l)
    # #limit vel
    # if v_r > 0.2 : 
    #     v_r = 0.2
    # elif v_r < -0.2:
    #     v_r = -0.2

    # if v_l > 0.2 : 
    #     v_l = 0.2
    # elif v_l < -0.2:
    #     v_l = -0.2

    # Update follower's position
    follower.update_odom(v_r, v_l, L)
    
    # Update leader circle and arrow
    leader_circle.center = (leader.x, leader.y)
    leader_arrow.remove()
    leader_arrow = ax.arrow(leader.x, leader.y, RADIUS * np.cos(leader.theta), RADIUS * np.sin(leader.theta), head_width=0.05, head_length=0.1, fc='b', ec='b')

    # Update follower circle and arrow
    follower_circle.center = (follower.x, follower.y)
    follower_arrow.remove()
    follower_arrow = ax.arrow(follower.x, follower.y, RADIUS * np.cos(follower.theta), RADIUS * np.sin(follower.theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

    return leader_circle, follower_circle, leader_arrow, follower_arrow

# Define the number of frames to ensure the animation stops after 3 seconds
num_frames = int(30 / dt)

ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames), init_func=init, blit=True, repeat=False, interval=dt*1000)
plt.show()
