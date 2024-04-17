import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

h = pd.read_csv('hockenheim/track_data.csv')
h_x = h['x_m']
h_y = h['y_m']

s = pd.read_csv('silverstone/track_data.csv')
s_x = s['x_m']
s_y = s['y_m']

h_arc_lengths = []
s_arc_lengths = []

h_bezier_points = []
s_bezier_points = []

h_indices = [53, 167, 178, 192, 257, 422, 509, 566, 586, 599, 606, 692, 757, 792, 806, 829, 856]
s_indices = [79, 123, 178, 209, 250, 390, 441, 505, 627, 722, 744, 777, 804, 838, 1016, 1103, 1120, 1156]

# Function to plot Bézier curves
def plot_bezier_curve(ax, control_points, color='b'):
    # Create a Path object from the Bézier points
    vertices = [(x, y) for x, y in control_points]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(vertices) - 1)
    path = Path(vertices, codes)

    # Plot the Bézier curve
    patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor=color)
    ax.add_patch(patch)


def calculate_arc_length(x, y):
    arc_length = 0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        ds = np.sqrt(dx**2 + dy**2)  # Euclidean distance formula
        arc_length += ds
    return arc_length

for i in h_indices:
    h_corner_x = [h_x[j] for j in range(i - 10, i + 11)]
    h_corner_y = [h_y[j] for j in range(i - 10, i + 11)]
    # plt.plot(h_corner_x, h_corner_y, color='black')
    arc_length = calculate_arc_length(h_corner_x, h_corner_y)
    h_arc_lengths.append(arc_length)

# print(h_arc_lengths)
# print(len(h_arc_lengths))

# plt.xlabel('Eastward distance (m)')
# plt.ylabel('Northward distance (m)')
# plt.title('Hockenheim')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')  # Set aspect ratio to equal
# plt.show()

for i in s_indices:
    s_corner_x = [s_x[j] for j in range(i - 10, i + 11)]
    s_corner_y = [s_y[j] for j in range(i - 10, i + 11)]
    # plt.plot(s_corner_x, s_corner_y, color='black')
    arc_length = calculate_arc_length(s_corner_x, s_corner_y)
    s_arc_lengths.append(arc_length)

# plt.xlabel('Eastward distance (m)')
# plt.ylabel('Northward distance (m)')
# plt.title('Silverstone')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')  # Set aspect ratio to equal
# plt.show()

# print("Arc lengths for Hockenheim:", h_arc_lengths)
# print("Arc lengths for Silverstone:", s_arc_lengths)

# Function to create lookup table
def create_lookup_table(x, y):
    arc_length = 0
    lut = [0]  # Initialize the lookup table with the starting point
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        ds = np.sqrt(dx**2 + dy**2)  # Euclidean distance formula
        arc_length += ds
        lut.append(arc_length)
    return lut

# Function to convert distance along track to parameter within segment
def dist_to_t(lut, distance):
    arc_length = lut[-1]
    n = len(lut)

    if 0 <= distance <= arc_length:
        for i in range(n - 1):
            if lut[i] <= distance <= lut[i + 1]:
                return remap(distance, lut[i], lut[i + 1], i / (n - 1), (i + 1) / (n - 1))
    return distance / arc_length


def calculate_p_1(p_t, p_0, p_2, t):
    p_1 = -37
    p_1_arr = np.array([])

    # p_t must be an array of all 100 / however many we can manage (21) for each corner based on the lookup table 
    # t must also be an array of all 100 / however many we can manage (21) for each corner based on the lookup table

    #p_0 = [x,y]
    #p_2 = [x,y]

    # const = const + const * var + const
    # (p_t - p_0 * sub1 - p_2 * sub3) / sub_2 = p_1[x]
    #[x] because we take the average of many possible values for p_1???

    for i in range(1, len(t) - 1):
        sub_1 = (t[i] ** 2) - (2 * t[i]) + 1
        sub_2 = (2 * t[i]) - (2 * (t[i] ** 2))
        sub_3 = (t[i] ** 2)

        term_1 = sub_1 * np.array(p_0) 
        term_3 = sub_3 * np.array(p_2)

        p_1_temp = (np.array(p_t[i]) - np.array(term_3) - np.array(term_1)) / (sub_2)
        p_1_arr = np.append(p_1_arr, p_1_temp)
        

    data_reshaped = p_1_arr.reshape(-1, 2)
    avg_x = np.mean(data_reshaped[:, 0])
    avg_y = np.mean(data_reshaped[:, 1])
    return [p_0, [avg_x, avg_y], p_2]

# Function to remap value from one range to another
def remap(value, from_start, from_end, to_start, to_end):
    return to_start + (value - from_start) * (to_end - to_start) / (from_end - from_start)

# create lookup tables (aka a cumulative distance table) (aka a )
h_luts = []
for i in h_indices:
    h_corner_x = [h_x[j] for j in range(i - 10, i + 11)]
    h_corner_y = [h_y[j] for j in range(i - 10, i + 11)]
    h_luts.append(create_lookup_table(h_corner_x, h_corner_y))

s_luts = []
for i in s_indices:
    s_corner_x = [s_x[j] for j in range(i - 10, i + 11)]
    s_corner_y = [s_y[j] for j in range(i - 10, i + 11)]
    s_luts.append(create_lookup_table(s_corner_x, s_corner_y))

h_t_val_tables = []
for h_lut in h_luts:
    for distance in h_lut:
        h_t_val_tables.append(dist_to_t(h_lut, distance))

s_t_val_tables = []
for s_lut in s_luts:
    for distance in s_lut:
        s_t_val_tables.append(dist_to_t(h_lut, distance))

# prepping blocks to calculate p1 point
for i in range(0, len(h_indices)):
    h_corner_x = [h_x[j] for j in range(h_indices[i] - 10, h_indices[i] + 11)]
    h_corner_y = [h_y[j] for j in range(h_indices[i] - 10, h_indices[i] + 11)]
    # print(h_corner_x)
    # print(h_corner_y)
    h_corner_pos = [[x, y] for x, y in zip(h_corner_x, h_corner_y)]
    t_vals = [h_t_val_tables[j] for j in range(i*21, (20 + i*21) + 1)]
    h_bezier_point = calculate_p_1(h_corner_pos, [h_corner_x[0], h_corner_y[0]], [h_corner_x[-1], h_corner_y[-1]], t_vals)
    h_bezier_points.append(h_bezier_point)


for i in range(0, len(s_indices)):
    s_corner_x = [s_x[j] for j in range(s_indices[i] - 10, s_indices[i] + 11)]
    s_corner_y = [s_y[j] for j in range(s_indices[i] - 10, s_indices[i] + 11)]
    s_corner_pos = [[x, y] for x, y in zip(s_corner_x, s_corner_y)]
    t_vals = [s_t_val_tables[j] for j in range(i*21, (20 + i*21) + 1)]
    s_bezier_point = calculate_p_1(s_corner_pos, [s_corner_x[0], s_corner_y[0]], [s_corner_x[-1], s_corner_y[-1]], t_vals)
    s_bezier_points.append(s_bezier_point)

# print(h_bezier_points)
# print(s_bezier_points)

def plot_track_with_bezier(track_x, track_y, bezier_points, label_points=False):
    plt.figure(figsize=(10, 6))
    
    # Plot track data
    plt.plot(track_x, track_y, color='black', label='Track')
    
    # Define 18 distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
              '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7']
    
    # Plot Bézier points
    for i, points in enumerate(bezier_points):
        color = colors[i % len(colors)]  # Cycle through the color list
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x, y, color=color, label=f'Bézier Points {i+1}')
        
        if label_points:
            for j, (x, y) in enumerate(points):
                plt.text(x, y, f'P{j}', fontsize=8, ha='center', va='bottom')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Track with Bézier Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

print("s_bezier_points")
print(s_bezier_points)
print("h_bezier_points")
print(h_bezier_points)


# Plot Hockenheim track with Bézier points
plot_track_with_bezier(h_x, h_y, h_bezier_points, label_points=True)

# Plot Silverstone track with Bézier points
plot_track_with_bezier(s_x, s_y, s_bezier_points, label_points=True)

