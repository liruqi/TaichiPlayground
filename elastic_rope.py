# Fixed head elastic string simulation
# Authored by Ruqi Li, liruqi@gmail.com

import math, os

import taichi as ti
import numpy as np
import cv2
from inspect import currentframe, getframeinfo

ti.init(arch=ti.gpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 0.001

# number of molecule
N = 200

# mass of a single molecule
mass = 100 / N / N 
init_string_length = 0.4
zero_force_distance = init_string_length / N

# init vel
init_vel = 0

# drag coefficient
cd = 0.001

# time-step size
h = 1e-1
# substepping
substepping = 10

# center of the screen
center = ti.Vector.field(2, ti.f32, ())
fixedPoint = ti.Vector.field(2, ti.f32, ())

# pos, vel and acceleration of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N+1)
vel = ti.Vector.field(2, ti.f32, N+1)
acceleration = ti.Vector.field(2, ti.f32, N+1)

@ti.func
def spring_acc(pv, i):
    d = pv.norm()
    #d = ti.sqrt(pv[0]*pv[0] + pv[1]*pv[1])
    zeroForceVector = pv * zero_force_distance / d
    #if i == 1:
    #    print(pv, d, zero_force_distance, zeroForceVector, (pv - zeroForceVector) / mass)
    return (pv - zeroForceVector) / mass

@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    fixedPoint[None] = [0.5, 0.8]
    pos[0] = fixedPoint[None]
    for i in range(N + 1):
        pos[i] = [fixedPoint[None][0] + i * init_string_length / N, fixedPoint[None][1]]
        vel[i] = [0, 0]
    
@ti.kernel
def compute_acceleration():
    # clear acceleration
    acceleration[0] = [0.0, 0.0]
    for i in range(1, N+1):
        acceleration[i] = [0.0, -G]

    # compute gravitational + spring acceleration
    for i in range(1, N+1):
        p = pos[i]
        leftVector = pos[i-1] - p
        leftAcc = spring_acc(leftVector, i)
        acceleration[i] += leftAcc
        if i+1 <= N:
            rightVector = pos[i+1] - p
            rightAcc = spring_acc(rightVector, i)
            acceleration[i] += rightAcc
        acceleration[i] -= cd * vel[i] * vel[i].norm()
    #print("acceleration 1:", acceleration[1])

@ti.kernel
def update():
    dt = h / substepping
    for i in range(1, N+1):
        #symplectic euler
        vel[i] += dt * acceleration[i]
        pos[i] += dt * vel[i]
    #print("pos 1:", pos[1])

gui = ti.GUI('Fixed head elastic rope', (1000, 1000))

initialize()
im = cv2.imread("res/galaxy.png")
print (im.shape)

result_dir = "./results"
video_manager = None
if os.path.exists(result_dir):
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

cnt = 0
cf = currentframe()

while gui.running:
    if video_manager and cnt > 1000:
        break
    cnt += 1
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]

    if not paused[None]:
        for i in range(substepping):
            compute_acceleration()
            update()

    gui.set_image(im)
    pos_np = pos.to_numpy()
    gui.lines(pos_np[:N],
        pos_np[1:],
        color=0xff66cc,
        radius=1)
    gui.circles(pos_np, color=0xffffff, radius=2)

    if video_manager:
        pixels_img = gui.get_image()
        video_manager.write_frame(pixels_img)
    else:
        gui.show()

if video_manager:
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

