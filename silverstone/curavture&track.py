# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# reads
data = pd.read_csv('silverstone/track_data.csv')
x = data['x_m']
y = data['y_m']

# calculate curvature in form (x'y'' - y'x'') / sqrt((x'^2 + y'^2) ^ 3)
dx_ds = np.gradient(x)
dy_ds = np.gradient(y)
d2x_ds2 = np.gradient(dx_ds)
d2y_ds2 = np.gradient(dy_ds)


print(np.gradient())

curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5

# plot local minima and maxima
local_min_indices = argrelextrema(curvature, np.less)[0]
local_max_indices = argrelextrema(curvature, np.greater)[0]

min_values = curvature[local_min_indices]
max_values = curvature[local_max_indices]

turn_indices = [79, 123, 178, 209, 250, 390, 441, 505, 627, 722, 744, 777, 804, 838, 1016, 1103, 1120, 1156]


# Plot curvature graph
plt.plot(curvature, label='Curvature')
# plt.scatter(local_min_indices, min_values, c='red', label='minima')
# plt.scatter(local_max_indices, max_values, c='green', label='maxima')
plt.scatter(turn_indices, curvature[turn_indices], c='orange', label='Turns')
plt.xlabel('Point along the track')
plt.ylabel('Curvature')
plt.title('Curvature Graph of Silverstone with turns highlighted')
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

# for index in turn_indices:
#     plt.scatter(x[index], y[index], color='green', marker='o', s=100)  # Increase marker size (s=100)

# for index in border_indices:
#     plt.scatter(x[index], y[index], color='red', marker='o', s=100)  # Increase marker size (s=100)

for index in turn_indices:
     plt.scatter(x[index], y[index], color='orange', marker='o', s=100)  # Increase marker size (s=100)


# Customize plot
plt.xlabel('Eastward distance (m)')
plt.ylabel('Northward distance (m)')
plt.title('Silverstone')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Set aspect ratio to equal
plt.show()



