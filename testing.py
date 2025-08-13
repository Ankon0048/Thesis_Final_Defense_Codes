import os
line_segment_output_dir = r"kaggle/output/cropped_outputs_line/"
dirs = os.listdir(line_segment_output_dir)
import shutil
for dir in dirs:
    shutil.rmtree(os.path.join(line_segment_output_dir,dir))