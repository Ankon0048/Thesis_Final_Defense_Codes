import cv2
import numpy as np
from pygam import LinearGAM
from bresenham import bresenham
from scipy.interpolate import interp1d
from numba import njit
import matplotlib.pyplot as plt

# === Configurable Parameters ===
n_splines = 6  # Fewer splines = faster fitting
threshold_value = 30  # For black pixel detection

# === Utility Functions ===

def divide_arc_length(X, Y, n):
    dx = np.diff(X)
    dy = np.diff(Y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    L = s[-1]
    Delta_L = L / n
    x_points = [X[0]]
    for i in range(1, n):
        target_length = i * Delta_L
        idx = np.searchsorted(s, target_length)
        x0, x1 = X[idx - 1], X[idx]
        s0, s1 = s[idx - 1], s[idx]
        x_interp = x0 + (target_length - s0) * (x1 - x0) / (s1 - s0)
        x_points.append(x_interp)
    x_points.append(X[-1])
    return x_points

@njit
def calculate_derivative(y_values):
    dy = np.zeros_like(y_values)
    dy[0] = y_values[1] - y_values[0]
    dy[-1] = y_values[-1] - y_values[-2]
    for i in range(1, len(y_values) - 1):
        dy[i] = (y_values[i + 1] - y_values[i - 1]) / 2
    return dy

def find_perpendicular_points(y_values, x_values, d):
    dy = calculate_derivative(np.array(y_values))
    perp_points = []
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        slope = dy[i]
        perp_slope = -1 / slope if slope != 0 else np.inf
        if np.isinf(perp_slope):
            pts = [(round(x), round(y - d)), (round(x), round(y + d))]
        else:
            c = y - perp_slope * x
            delta = d / np.sqrt(1 + perp_slope ** 2)
            x1, x2 = x + delta, x - delta
            y1, y2 = perp_slope * x1 + c, perp_slope * x2 + c
            pts = [(round(x1), round(y1)), (round(x2), round(y2))]
        perp_points.append(pts)
    return perp_points

def find_distance_d(X, y, X_new, y_hat, step):
    d, iteration, max_iter = 0, 0, 1000
    while iteration < max_iter:
        upper = y_hat + d
        lower = y_hat - d
        all_covered = all((y[i] >= lower[np.argmin(np.abs(X_new - X[i]))]) and 
                          (y[i] <= upper[np.argmin(np.abs(X_new - X[i]))])
                          for i in range(len(X_new)))
        if all_covered:
            break
        d += step
        iteration += 1
    return int(np.ceil(2 * d))

def uncurve_text_tight(image, output_path, n_splines=6, show_plot=False, arc_equal=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    black_pixels = np.column_stack(np.where(thresh <= threshold_value))
    X = black_pixels[:, 1].reshape(-1, 1)
    y = black_pixels[:, 0]
    left_x = np.min(black_pixels[:, 1])
    right_x = np.max(black_pixels[:, 1])

    gam = LinearGAM(n_splines=n_splines)
    gam.fit(X, y)

    if not arc_equal:
        X_new = np.linspace(left_x, right_x, num=right_x - left_x)
    else:
        X_dense = np.linspace(left_x, right_x, num=right_x - left_x)
        Y_dense = gam.predict(X_dense)
        X_new = divide_arc_length(X_dense, Y_dense, right_x - left_x)

    y_hat = gam.predict(X_new)

    if show_plot:
        plt.imshow(thresh, cmap='gray')
        plt.plot(X_new, y_hat, color='red')
        plt.axis('off')
        plt.show()

    d = find_distance_d(X.flatten(), y, X_new, y_hat, step=0.5)
    dewarp_image = np.full(((2 * d + 1), len(X_new)), 255, dtype=np.uint8)

    perp_points = find_perpendicular_points(y_hat, X_new, d)
    for i, points in enumerate(perp_points):
        x1, y1 = points[0]
        x2, y2 = points[1]
        if y1 > y2:
            x1, x2, y1, y2 = x2, x1, y2, y1

        bres_list = list(bresenham(x1, y1, x2, y2))
        pixels = []
        for x, y in bres_list:
            x = np.clip(x, 0, thresh.shape[1] - 1)
            y = np.clip(y, 0, thresh.shape[0] - 1)
            pixels.append(thresh[y, x])

        pixel_array = np.array(pixels).reshape(-1, 1)
        resized_column = cv2.resize(pixel_array, (1, 2 * d + 1), interpolation=cv2.INTER_LINEAR).flatten()
        dewarp_image[:, i] = resized_column

    if show_plot:
        plt.imshow(dewarp_image, cmap='gray')
        plt.axis('off')
        plt.show()

    print(f'output path: {output_path}')
    print(cv2.imwrite(output_path, dewarp_image))
