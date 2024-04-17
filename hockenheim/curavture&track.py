# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# reads
data = pd.read_csv('hockenheim/track_data.csv')
x = data['x_m']
y = data['y_m']

# calculate curvature in form (x'y'' - y'x'') / sqrt((x'^2 + y'^2) ^ 3)
dx_ds = np.gradient(x)
dy_ds = np.gradient(y)
d2x_ds2 = np.gradient(dx_ds)
d2y_ds2 = np.gradient(dy_ds)

curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5

# plot local minima and maxima
local_min_indices = argrelextrema(curvature, np.less)[0]
local_max_indices = argrelextrema(curvature, np.greater)[0]

min_values = curvature[local_min_indices]
max_values = curvature[local_max_indices]

turn_indices = [53, 167, 178, 192, 257, 422, 509, 566, 586, 599, 606, 692, 757, 792, 806, 829, 856]
border_indices = [38, 68, 152, 182, 163, 193, 177, 207, 242, 272, 407, 437, 494, 524, 551, 581, 571, 601, 584, 614, 591, 621, 677, 707, 742, 772, 777, 807, 791, 821, 814, 844, 841 ,871]

# Plot curvature graph
plt.plot(curvature, label='Curvature')
# plt.scatter(local_min_indices, min_values, c='red', label='minima')
# plt.scatter(local_max_indices, max_values, c='green', label='maxima')
plt.scatter(turn_indices, curvature[turn_indices], c='orange', label='Turns')
plt.xlabel('Point along the track')
plt.ylabel('Curvature')
plt.title('Curvature Graph of Hockenheim with turns highlighted')
plt.legend()
plt.grid(True)
plt.show()





# Plot track boundaries
x = data['x_m']
y = data['y_m']
w_tr_right = data['w_tr_right_m']
w_tr_left = data['w_tr_left_m']

plt.plot(x, y, label='Centerline', color='black')

for i in range(len(x)):
    plt.plot([x[i] + w_tr_right[i], x[i]], [y[i], y[i]], color='blue')  # Plot boundary to the right
    plt.plot([x[i], x[i] - w_tr_left[i]], [y[i], y[i]], color='red')    # Plot boundary to the left

# Calculate tangent direction at (0,0)
tangent_angle = np.arctan2(dy_ds[0], dx_ds[0])

# Plot the arrow
plt.arrow(0, 0, 1*np.cos(tangent_angle), 1*np.sin(tangent_angle),
          head_width=45, head_length=60, fc='green', ec='green', zorder=10)

for index in turn_indices:
     plt.scatter(x[index], y[index], color='orange', marker='o', s=100)

# for index in local_min_indices:
#      plt.scatter(x[index], y[index], color='red', marker='o', s=100)  # Increase marker size (s=100)
# for index in local_max_indices:
#     plt.scatter(x[index], y[index], color='green', marker='o', s=100)  # Increase marker size (s=100)

# Customize plot
# plt.xlabel('Eastward distance (m)')
# plt.ylabel('Northward distance (m)')
# plt.title('Hockhenheim')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')  # Set aspect ratio to equal
# plt.show()



