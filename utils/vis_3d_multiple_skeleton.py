
import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

ax = plt.axes(projection='3d')

namafile = 'vis_kps.npy'
header1 = "X Label"
header2 = "Z Label"
header3 = "Y Label"

# index = count()

 # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
cmap = plt.get_cmap('rainbow')
kps_lines = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
colors = [np.array((c[2], c[1], c[0])) for c in colors]

kpt_3d = np.load(namafile)


def vis_3d_multiple_skeleton(i):
    plt.cla()

    kpt_3d_vis = np.ones_like(kpt_3d)

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    
    
    # plt.pause(0.0001) #Note this correction
    # plt.show()


# def animate(i):
#     vis_kps = np.load("vis_kps.npy")
   

#     plt.cla()

#     ax.plot3D(x, y, z, 'red')


#     #plt.legend(loc='upper left')
#     #plt.tight_layout()


ani = FuncAnimation(plt.gcf(), vis_3d_multiple_skeleton, interval=1)

plt.tight_layout()
plt.show()
