import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
dt = 0.1  # time step
L = 0.6  # distance between robots
RADIUS = 0.2  # radius of the robot
vmax = 0.2
time_step = 0
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
        self.x += dc * np.cos(self.theta)
        self.y += dc * np.sin(self.theta)

# Initialize robots
leader = Robot(0, 0, 0)
follower1 = Robot(-L, 0, np.deg2rad(0))
follower2 = Robot(-2 * L, 0, np.deg2rad(0))

# Initialize PID controllers
pid_vx = PID(15, 1.5, 0.2)
pid_vy = PID(15, 1.5, 0.2)
pid_w = PID(15, 1.5, 0.1)

# Store trajectories
leader_trajectory = {'x': [], 'y': [], 'theta': []}
follower1_trajectory = {'x': [], 'y': [], 'theta': []}
follower2_trajectory = {'x': [], 'y': [], 'theta': []}

# Simulation
fig, ax = plt.subplots()
leader_circle = plt.Circle((leader.x, leader.y), RADIUS, fill=False, edgecolor='b')
follower1_circle = plt.Circle((follower1.x, follower1.y), RADIUS, fill=False, edgecolor='r')
follower2_circle = plt.Circle((follower2.x, follower2.y), RADIUS, fill=False, edgecolor='k')

leader_arrow = ax.arrow(leader.x, leader.y, RADIUS * np.cos(leader.theta), RADIUS * np.sin(leader.theta), head_width=0.05, head_length=0.1, fc='b', ec='b')
follower1_arrow = ax.arrow(follower1.x, follower1.y, RADIUS * np.cos(follower1.theta), RADIUS * np.sin(follower1.theta), head_width=0.05, head_length=0.1, fc='r', ec='r')
follower2_arrow = ax.arrow(follower2.x, follower2.y, RADIUS * np.cos(follower2.theta), RADIUS * np.sin(follower2.theta), head_width=0.05, head_length=0.1, fc='k', ec='k')

ax.add_patch(leader_circle)
ax.add_patch(follower1_circle)
ax.add_patch(follower2_circle)

def limit_velocity(v_r, v_l):
    if v_r > vmax:
        v_r = vmax
    elif v_r < -vmax:
        v_r = -vmax

    if v_l > vmax:
        v_l = vmax
    elif v_l < -vmax:
        v_l = -vmax
    return v_r, v_l

def init():
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    ax.grid(True)
    return leader_circle, follower1_circle, follower2_circle, leader_arrow, follower1_arrow, follower2_arrow

def update(frame):
    global leader, follower1, follower2, leader_circle, follower1_circle, follower2_circle, leader_arrow, follower1_arrow, follower2_arrow,time_step
    end_leader_flag = 0
    time_step+=1
    print(time_step)

    # Leader's control (move in the x direction with constant speed)
    if (leader.x <= 1) and (leader.y <= 0):
        leader_v = 0.1
        leader_thetad = np.deg2rad(0)
    elif (leader.x <= 2) and (leader.y <= 2):
        leader_v = 0.1
        leader_thetad = np.deg2rad(45)
    elif (leader.x <= 4) and (leader.y <= 2):
        leader_v = 0.1
        leader_thetad = np.deg2rad(0)
    else:
        leader_v = 0.0
        leader_thetad = np.deg2rad(0)
        end_leader_flag = 1

    w_error_leader = leader_thetad - leader.theta
    leader_w = pid_w.update(w_error_leader)
    vr_leader = leader_v + (leader_w * 0.2 / 2)
    vl_leader = leader_v - (leader_w * 0.2 / 2)
    vr_leader, vl_leader = limit_velocity(vr_leader, vl_leader)
    if end_leader_flag == 1:
        vr_leader = vl_leader = 0

    leader.update_odom(vr_leader, vl_leader, leader_w)

    # Calculate the distance and angle error for the follower
    x1_d = leader.x - L * np.cos(leader.theta)
    y1_d = leader.y - L * np.sin(leader.theta)
    x2_d = leader.x - 2 * L * np.cos(leader.theta)
    y2_d = leader.y - 2 * L * np.sin(leader.theta)

    x1_error = x1_d - follower1.x
    y1_error = y1_d - follower1.y
    x2_error = x2_d - follower2.x
    y2_error = y2_d - follower2.y

    vx1 = pid_vx.update(x1_error)
    vy1 = pid_vy.update(y1_error)
    vx2 = pid_vx.update(x2_error)
    vy2 = pid_vy.update(y2_error)

    v1 = vx1 * np.cos(follower1.theta) + vy1 * np.sin(follower1.theta)
    w1 = (vx1 * np.cos(np.pi * 1.5 - follower1.theta) - vy1 * np.sin(np.pi * 1.5 - follower1.theta)) / 0.0325
    v2 = vx2 * np.cos(follower2.theta) + vy2 * np.sin(follower2.theta)
    w2 = (vx2 * np.cos(np.pi * 1.5 - follower2.theta) - vy2 * np.sin(np.pi * 1.5 - follower2.theta)) / 0.0325

    v_r1 = v1 + (w1 * 0.2 / 2)
    v_l1 = v1 - (w1 * 0.2 / 2)
    v_r2 = v2 + (w2 * 0.2 / 2)
    v_l2 = v2 - (w2 * 0.2 / 2)

    v_r1, v_l1 = limit_velocity(v_r1, v_l1)
    v_r2, v_l2 = limit_velocity(v_r2, v_l2)

    if np.sqrt((leader.x - follower1.x) ** 2 + (leader.y - follower1.y) ** 2) <= L - 0.005:
        v_r1 = v_l1 = 0

    if np.sqrt((follower1.x - follower2.x) ** 2 + (follower1.y - follower2.y) ** 2) <= L - 0.005:
        v_r2 = v_l2 = 0

    follower1.update_odom(v_r1, v_l1, w1)
    follower2.update_odom(v_r2, v_l2, w2)

    # Store trajectories
    leader_trajectory['x'].append(leader.x)
    leader_trajectory['y'].append(leader.y)
    leader_trajectory['theta'].append(np.rad2deg(leader.theta))

    follower1_trajectory['x'].append(follower1.x)
    follower1_trajectory['y'].append(follower1.y)
    follower1_trajectory['theta'].append(np.rad2deg(follower1.theta))

    follower2_trajectory['x'].append(follower2.x)
    follower2_trajectory['y'].append(follower2.y)
    follower2_trajectory['theta'].append(np.rad2deg(follower2.theta))

    # Update leader circle and arrow
    leader_circle.center = (leader.x, leader.y)
    leader_arrow.remove()
    leader_arrow = ax.arrow(leader.x, leader.y, RADIUS * np.cos(leader.theta), RADIUS * np.sin(leader.theta), head_width=0.05, head_length=0.1, fc='b', ec='b')

    # Update follower circle and arrow
    follower1_circle.center = (follower1.x, follower1.y)
    follower1_arrow.remove()
    follower1_arrow = ax.arrow(follower1.x, follower1.y, RADIUS * np.cos(follower1.theta), RADIUS * np.sin(follower1.theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

    follower2_circle.center = (follower2.x, follower2.y)
    follower2_arrow.remove()
    follower2_arrow = ax.arrow(follower2.x, follower2.y, RADIUS * np.cos(follower2.theta), RADIUS * np.sin(follower2.theta), head_width=0.05, head_length=0.1, fc='k', ec='k')

    return leader_circle, follower1_circle, follower2_circle, leader_arrow, follower1_arrow, follower2_arrow

# Define the number of frames to ensure the animation stops after 3 seconds
num_frames = int(100 / dt)

ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames), init_func=init, blit=True, repeat=False, interval=dt*1000)
plt.show()

print("end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{}".format(time_step))
# Plotting trajectories
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot x trajectories
axs[0].plot(leader_trajectory['x'], label='Leader')
axs[0].plot(follower1_trajectory['x'], label='Follower 1')
axs[0].plot(follower2_trajectory['x'], label='Follower 2')
axs[0].set_ylabel('x')
axs[0].legend()

# Plot y trajectories
axs[1].plot(leader_trajectory['y'], label='Leader')
axs[1].plot(follower1_trajectory['y'], label='Follower 1')
axs[1].plot(follower2_trajectory['y'], label='Follower 2')
axs[1].set_ylabel('y')
axs[1].legend()

# Plot theta trajectories
axs[2].plot(leader_trajectory['theta'], label='Leader')
axs[2].plot(follower1_trajectory['theta'], label='Follower 1')
axs[2].plot(follower2_trajectory['theta'], label='Follower 2')
axs[2].set_ylabel('theta')
axs[2].legend()

plt.xlabel('Time step')
plt.show()
