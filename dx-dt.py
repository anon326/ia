import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h = pd.read_csv('hockenheim/track_data.csv')
h_x = h['x_m']
h_y = h['y_m']

s = pd.read_csv('silverstone/track_data.csv')
s_x = s['x_m']
s_y = s['y_m']

h_bezier_points = [[[-95.234657, 190.166437], [-127.52713988597698, 235.13952969502958], [-98.814549, 281.294085]], [[203.714949, 637.263159], [244.85861601376752, 691.7342973192968], [275.023084, 637.996462]], [[245.796844, 669.251363], [288.9729421075164, 642.6518154140884], [280.388068, 583.516813]], [[279.635508, 613.750383], [273.7390720526853, 558.9093129149645], [316.568633, 526.762614]], [[514.67729, 420.590035], [561.188339548342, 402.0080797715042], [609.960094, 390.341691]], [[1307.768592, 522.317825], [1370.4666487791178, 525.9045507146294], [1309.406508, 482.315289]], [[1028.630297, 299.306016], [987.2117774139392, 266.19899889941837], [935.301379, 278.093161]], [[752.340434, 306.390347], [691.4621130800703, 319.5647687799946], [699.931433, 255.284901]], [[699.931433, 255.284901], [706.5460026076322, 205.23416218448187], [737.522228, 164.734807]], [[718.516658, 193.765843], [756.8266820497405, 150.0699722754482], [720.090475, 105.036277]], [[737.522228, 164.734807], [743.838071545175, 110.86302940220831], [692.191736, 84.214546]], [[414.836044, -94.950251], [366.2305049188126, -122.86698276386558], [324.100623, -83.143228]], [[180.382828, 90.182644], [131.62107835899337, 132.83337493166488], [104.324979, 75.517261]], [[120.703128, 2.860747], [134.5881796662442, -48.78715226359808], [187.493079, -63.249069]], [[159.859801, -51.879466], [209.1240243624033, -67.73027966775956], [235.761084, -112.740799]], [[244.368323, -124.998489], [276.6441447254328, -174.48026009651684], [222.767589, -205.374724]], [[195.170426, -226.909707], [148.09651696984776, -260.7542040635704], [108.182188, -216.631488]]]


s_bezier_points = [[[205.345402, 279.267521], [230.3285895914089, 327.7560388233531], [283.513254, 331.687958]], [[403.318709, 326.137475], [454.4158790581913, 319.4913243581458], [499.850612, 344.244545]], [[634.902193, 455.274633], [682.6880247050491, 503.2668473717044], [706.529154, 445.854824]], [[723.917579, 393.680633], [749.0465477369055, 330.5878865139104], [783.464811, 389.181506]], [[807.372706, 491.018256], [814.5950291860848, 544.4523550983407], [768.609904, 575.746352]], [[312.708143, 965.839457], [268.0485207461415, 1000.5521968924542], [219.969175, 972.84014]], [[172.255643, 832.429357], [116.56302909898925, 811.9297244730906], [95.229592, 869.960079]], [[182.445954, 1068.022411], [213.47686948526737, 1108.4085705099778], [258.552288, 1131.081527]], [[761.641902, 1179.219262], [809.9838413650104, 1158.7171057020714], [823.263921, 1106.029473]], [[884.121878, 737.566632], [884.5059921908284, 686.9378427293013], [909.050957, 641.873306]], [[914.11595, 633.283317], [949.024023876371, 591.3815314626914], [927.030071, 540.70791]], [[907.56572, 478.787013], [892.5374264171705, 427.4004913262674], [924.502083, 384.25164]], [[946.987057, 357.515196], [976.867523680844, 309.83305766478753], [940.994102, 265.582483]], [[884.648112, 224.541458], [838.4888637714625, 201.33643529240365], [817.906522, 153.455651]], [[429.105721, -526.521092], [376.94909786565233, -543.3687567149888], [337.107716, -505.620305]], [[134.051911, -242.168203], [99.68947986855298, -187.10897250723778], [57.630959, -227.864964]], [[68.417857, -217.240894], [25.639399146403672, -260.2535825599363], [-12.486652, -209.439718]], [[-51.152437, -140.015353], [-70.1448597879667, -89.68887807512326], [-31.738762, -49.109186]]]

h_indices = [53, 167, 178, 192, 257, 422, 509, 566, 586, 599, 606, 692, 757, 792, 806, 829, 856]
s_indices = [79, 123, 178, 209, 250, 390, 441, 505, 627, 722, 744, 777, 804, 838, 1016, 1103, 1120, 1156]

# Function to calculate the Bézier curve
def bezier_curve(t, P0, P1, P2):
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)

    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2

def bezier_curve_derivative(t, P0, P1, P2):
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)

    return (2 * t-2) * P0 + (2-4 * t) * P1 + 2 * t * P2

def bezier_curve_second_derivative(P0, P1, P2):
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)

    return 2 * P0 - 4 * P1 + 2 * P2

def calculate_average_curvature(x_values, y_values):
    # Calculate the first derivative
    dx_ds = np.gradient(x_values)
    dy_ds = np.gradient(y_values)
    
    # Calculate the second derivative
    d2x_ds2 = np.gradient(dx_ds)
    d2y_ds2 = np.gradient(dy_ds)
    
    # Calculate curvature
    curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5
    
    # Average curvature
    average_curvature = np.mean(np.abs(curvature))
    
    return average_curvature

def approximate_t_for_arc_length(P_0, P_1, P_2):
    t_values = np.linspace(0, 1, 101)  # 101 points for 100 intervals
    arc_lengths = [np.linalg.norm(bezier_curve(t, P_0, P_1, P_2) - bezier_curve(t_prev, P_0, P_1, P_2)) for t, t_prev in zip(t_values[1:], t_values[:-1])]
    total_arc_length = sum(arc_lengths)
    
    # Calculate cumulative arc lengths
    cumulative_arc_lengths = np.cumsum(arc_lengths)
    
    # Approximate t value
    target_length = total_arc_length / 2  # Assuming we want to find the t value at half the total arc length
    t_index = np.argmax(cumulative_arc_lengths >= target_length)
    t_approx = t_values[t_index]
    
    return t_approx


# Generate values for parameter t
t_values = np.linspace(0, 1, 100)

def get_actual_points_h(i):
   h_corner_x = [h_x[j] for j in range(i - 10, i + 11)]
   h_corner_y = [h_y[j] for j in range(i - 10, i + 11)]

   return [[h_corner_x[i], h_corner_y[i]] for i in range(len(h_corner_x))]

def get_actual_points_s(i):
   s_corner_x = [s_x[j] for j in range(i - 10, i + 11)]
   s_corner_y = [s_y[j] for j in range(i - 10, i + 11)]

   return [[s_corner_x[i], s_corner_y[i]] for i in range(len(s_corner_x))]

for i in range(0, 17):
 h_bezier_curve_points = np.array([bezier_curve(t, h_bezier_points[i][0], h_bezier_points[i][1], h_bezier_points[i][2]) for t in t_values])
 h_bezier_curve_points_x, h_bezier_curve_points_y = zip(*h_bezier_curve_points)
 
 h_bezier_derivative_curve_points = np.array([bezier_curve_derivative(t, h_bezier_points[i][0], h_bezier_points[i][1], h_bezier_points[i][2]) for t in t_values])
 h_bezier_derivative_curve_points_x, h_bezier_derivative_curve_points_y = zip(*h_bezier_derivative_curve_points)






 
#  plt.figure(figsize=(12, 6))

#  # Plot bezier points
#  plt.subplot(1, 2, 1)
#  plt.plot(h_bezier_curve_points_x, h_bezier_curve_points_y, label='Bézier Points', marker='o', color='blue')
#  plt.xlabel('X')
#  plt.ylabel('Y')
#  plt.title('Bézier Points')

#  # Plot actual points
#  plt.subplot(1, 2, 2)
#  plt.plot(h_bezier_derivative_curve_points_x, h_bezier_derivative_curve_points_y, label='Actual Points')
#  plt.xlabel('X')
#  plt.ylabel('Y')
#  plt.title('Actual Points')

#  plt.suptitle("Turn number " + f'{i + 1}' + " Hockenheim")

#  plt.tight_layout()
#  plt.show()

for i in range(0, 18):
 s_bezier_curve_points = np.array([bezier_curve(t, s_bezier_points[i][0], s_bezier_points[i][1], s_bezier_points[i][2]) for t in t_values])
 s_bezier_curve_points_x, s_bezier_curve_points_y = zip(*s_bezier_curve_points)
 
 s_bezier_derivative_curve_points = np.array([bezier_curve_derivative(t, s_bezier_points[i][0], s_bezier_points[i][1], s_bezier_points[i][2]) for t in t_values])
 s_bezier_derivative_curve_points_x, s_bezier_derivative_curve_points_y = zip(*s_bezier_derivative_curve_points)
 
 s_t_approx = approximate_t_for_arc_length(s_bezier_points[i][0], s_bezier_points[i][1], s_bezier_points[i][2])
 print(s_t_approx)

 
#  print(bezier_curve_second_derivative(s_bezier_points[i][0], s_bezier_points[i][1], s_bezier_points[i][2]))

#  print(bezier_curve_second_derivative(s_bezier_points[i][0], s_bezier_points[i][1], s_bezier_points[i][2]))
 
#  print(abs(sum((s_bezier_derivative_curve_points_y)) / len(s_bezier_derivative_curve_points_y)))


#  plt.figure(figsize=(12, 6))

#  # Plot bezier points
#  plt.subplot(1, 2, 1)
#  plt.plot(s_bezier_curve_points_x, s_bezier_curve_points_y, label='Bézier Points', marker='o', color='blue')
#  plt.xlabel('X')
#  plt.ylabel('Y')
#  plt.title('Bézier Points')

#  # Plot actual points
#  plt.subplot(1, 2, 2)
#  plt.plot(s_bezier_derivative_curve_points_x, s_bezier_derivative_curve_points_y, label='Derivative of Bézier Curve')
#  plt.xlabel('X')
#  plt.ylabel('Y')
#  plt.title('Actual Points')

#  plt.suptitle("Turn number " + f'{i + 1}' + " - Silverstone")

#  plt.tight_layout()
#  plt.show()


# print(h_bezier_derivative_curve_points)
# print(s_bezier_derivative_curve_points)
