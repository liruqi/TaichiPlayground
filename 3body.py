# Authored by Tiantian Liu, Taichi Graphics.
import math, os

import taichi as ti
import numpy as np
import cv2

ti.init(arch=ti.cpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1

# number of planets
N = 3
mass = ti.field(ti.i64, (N))

# for demonstration only, actual value: 333000
sun_mass_related_to_earth = 10
  
# unit mass
# m = 1
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = ti.field(ti.f32, (N))
# init vel
init_vel = 100

# time-step size
h = 1e-5
# substepping
substepping = 10

# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and acceleration of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
acceleration = ti.Vector.field(2, ti.f32, N)


@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    mass[0] = sun_mass_related_to_earth * 81
    mass[1] = 81
    mass[2] = 1

    planet_radius[0] = 6.0 * 9 
    planet_radius[1] = 6.0
    planet_radius[2] = 2.0
 
    # avoid Milky Way escaping canvas
    vel[0] = [- init_vel * 0.3 * 0.2 / sun_mass_related_to_earth, - init_vel * 0.4 * 0.85 / sun_mass_related_to_earth]
    vel[1] = [0, init_vel * 0.4]
    vel[2] = vel[1] + [init_vel * 0.8, 0]

    pos[0] = center[None] 
    pos[1] = center[None] + [0.8 * galaxy_size, 0]
    pos[2] = pos[1] + [0, (0.8 / 20) * galaxy_size]
    
@ti.kernel
def compute_acceleration():

    # clear acceleration
    for i in range(N):
        acceleration[i] = [0.0, 0.0]

    # compute gravitational acceleration
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                r = diff.norm(1e-5)

                # gravitational acceleration -(GMm / r^2) * (diff/r) for i
                f = -G * mass[j] * (1.0 / r)**3 * diff

                # assign to each particle
                acceleration[i] += f


@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        #symplectic euler
        vel[i] += dt * acceleration[i]
        pos[i] += dt * vel[i]
    print(pos[1])

gui = ti.GUI('3-body problem', (1000, 1000))

initialize()
im = cv2.imread("res/galaxy.png")
print (im.shape)

result_dir = "./results"
video_manager = None
if os.path.exists(result_dir):
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

cnt = 0
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
    gui.circles(pos.to_numpy(), color=np.array([0xea5a3e, 0x00ecff, 0xffffff]), radius=planet_radius.to_numpy())

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

