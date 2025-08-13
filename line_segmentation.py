# Update your base directories
base_input_dir = r"kaggle/output/cropped_outputs"
base_output_printed = r"kaggle/output/cropped_outputs_line"
base_graph_folder = r"kaggle/output/line_graphs"

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import binary_closing

def auto_savgol_smooth(profile, polyorder=2, spacing_factor=None,
                       plot=True, plot_title="", save_path=None,
                       show_thresholds=False, high_thresh=None, low_thresh=None):
    
    peaks, _ = find_peaks(profile, distance=8)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected to estimate line spacing.")

    diffs = np.diff(peaks)
    eps = 1e-9
    weights = 1.0 / (diffs + eps)
    avg_spacing = int(np.round(np.sum(weights * diffs) / np.sum(weights)))

    # Dynamically estimate spacing_factor if not provided
    if spacing_factor is None:
        spacing_factor = min(max(1.2, avg_spacing / 20), 2.0)

    window_length = int(spacing_factor * avg_spacing)
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, polyorder + 4)
    window_length = min(window_length,
                        len(profile) - 1 if len(profile) % 2 else len(profile) - 2)

    smoothed = savgol_filter(profile, window_length=window_length, polyorder=polyorder)

    if plot:
        fig = plt.figure(figsize=(14, 5))
        plt.plot(profile, label="Original", color="orange", alpha=0.6)
        plt.plot(smoothed, label=f"Smoothed (window={window_length})", color="blue")
        plt.plot(peaks, profile[peaks], "rx", label="Detected Peaks")

        if show_thresholds:
            if high_thresh is not None:
                plt.axhline(y=high_thresh, color="red", linestyle="--", label=f"High Thresh = {high_thresh:.2f}")
            if low_thresh is not None:
                plt.axhline(y=low_thresh, color="green", linestyle="--", label=f"Low Thresh = {low_thresh:.2f}")

        plt.title(plot_title or "Savitzky-Golay smoothing")
        plt.xlabel("Row Index")
        plt.ylabel("Sum of Pixel Intensities")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        plt.close(fig)

    return smoothed, spacing_factor



def calculate_projection_profile_and_crop_lines_with_lines(image_path, folder_name):
    base_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(base_name)[0]

    subfolder_graph = os.path.join(base_graph_folder, folder_name)
    os.makedirs(subfolder_graph, exist_ok=True)
    output_path = os.path.join(subfolder_graph, f"{base_name}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_projection = np.sum(binary_image, axis=1)

    smoothed, spacing_factor = auto_savgol_smooth(
        horizontal_projection,
        save_path=output_path,
        plot=True,
        show_thresholds=True
    )

    # === Dynamic Thresholds ===
    Q1 = np.percentile(smoothed, 25)
    Q3 = np.percentile(smoothed, 75)
    IQR = Q3 - Q1
    mean_val = np.mean(smoothed)
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)

    iqr_low = Q1 + 0.2 * IQR
    iqr_high = iqr_low + 0.2 * IQR
    mean_low = mean_val * 0.25
    mean_high = mean_val * 0.5
    scaled_low = min_val + 0.1 * (max_val - min_val)
    scaled_high = min_val + 0.3 * (max_val - min_val)

    low_thresh = np.median([iqr_low, mean_low, scaled_low])
    high_thresh = np.median([iqr_high, mean_high, scaled_high])

    # Re-plot with thresholds
    smoothed, _ = auto_savgol_smooth(
        horizontal_projection,
        spacing_factor=spacing_factor,
        save_path=output_path,
        plot=True,
        show_thresholds=True,
        high_thresh=high_thresh,
        low_thresh=low_thresh
    )

    # === Line Detection with Relaxed High Threshold at Bottom ===
    line_ranges = []
    is_in_line = False
    relaxed_zone = int(0.8 * len(smoothed))

    for row, value in enumerate(smoothed):
        current_high = high_thresh
        if row > relaxed_zone:
            current_high = high_thresh * 0.65  # relax threshold in bottom zone

        if value > current_high and not is_in_line:
            start_row = row
            is_in_line = True
        elif value < low_thresh and is_in_line:
            end_row = row
            line_ranges.append((start_row, end_row))
            is_in_line = False

    if is_in_line:
        line_ranges.append((start_row, len(smoothed)))

    # === Fallback: Recover Missed Final Line ===
    last_line_margin = int(len(smoothed) * 0.17)
    end_threshold = len(smoothed) - last_line_margin
    last_part_vals = smoothed[-last_line_margin:]

    if all(end < end_threshold for _, end in line_ranges):
        if np.max(last_part_vals) > low_thresh:
            fallback_start = end_threshold
            line_ranges.append((fallback_start, len(smoothed)))

    # === Refine Borders ===
    if line_ranges:
        line_ranges[0] = (max(0, line_ranges[0][0] - 5), line_ranges[0][1])
        line_ranges[-1] = (line_ranges[-1][0], min(image.shape[0], line_ranges[-1][1] + 5))

    for i in range(1, len(line_ranges)):
        temp = (line_ranges[i - 1][1] + line_ranges[i][0]) // 2
        line_ranges[i - 1] = (line_ranges[i - 1][0], temp)
        line_ranges[i] = (temp, line_ranges[i][1])

    line_ranges = sorted(line_ranges, key=lambda x: x[0])

    # === Save Cropped Lines ===
    subfolder_output = os.path.join(base_output_printed, folder_name, image_name_no_ext)
    os.makedirs(subfolder_output, exist_ok=True)

    for idx, (start, end) in enumerate(line_ranges, 1):
        cropped_line = image[start:end, :]
        save_path = os.path.join(subfolder_output, f"{idx}.png")
        print(f'save_path = {save_path}')
        cv2.imwrite(save_path, cropped_line)