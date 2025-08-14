import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk, ImageOps
import os
import threading
from queue import Queue
from model_module import Models  # your custom module
import shutil

# CONSTANT PATHS
yolo_output_dir = r"kaggle/output/"
showing_yolo_results_dir = r"kaggle/output/cropped_outputs/"
line_segment_output_dir = r"kaggle/output/cropped_outputs_line/"
directory = "kaggle/output/cropped_outputs"
yolo_images = []

selected_image_path = None
line_segment_images = []
models = None

def refresh_ui():
    global selected_image_path, line_segment_images

    # Clear selected image path
    selected_image_path = None

    # Clear displayed images and labels
    image_label.config(image=None)
    image_label.image = None

    yolo_output_image.config(image=None)
    yolo_output_image.image = None

    path_label.config(text="No image selected")

    status_label.config(text="UI refreshed. Models are still loaded.", fg="blue")

    # Clear line segment images list (keep references)
    line_segment_images.clear()

    # Delete all files in YOLO output directory
    if os.path.exists(showing_yolo_results_dir):
        dirs = os.listdir(showing_yolo_results_dir)
        for dir in dirs:
            # print(f'yolo directory {os.path.join(showing_yolo_results_dir,dir)}')
            shutil.rmtree(os.path.join(showing_yolo_results_dir,dir))

    # Delete all files in line segmentation output directory recursively
    if os.path.exists(line_segment_output_dir):
        dirs = os.listdir(line_segment_output_dir)
        for dir in dirs:
            # print(f'line segment directory {os.path.join(line_segment_output_dir,dir)}')
            shutil.rmtree(os.path.join(line_segment_output_dir,dir))

def add_black_border(pil_img, border_size=5):
    return ImageOps.expand(pil_img, border=border_size, fill="black")

def load_models_background():
    global models
    # This runs in background thread - no GUI updates here
    models = Models()
    # After loading, update status in main thread
    root.after(0, lambda: status_label.config(text="✅ Models loaded successfully!"))

def generate_ocr_result():
    global selected_image_path

    if not selected_image_path or not os.path.isdir(selected_image_path):
        status_label.config(text="⚠ Please select a valid folder.")
        return

    # Create result display window
    ocr_window = Toplevel(root)
    ocr_window.title("OCR Results")
    ocr_window.geometry("600x400")

    # Scrollable text box
    text_area = tk.Text(ocr_window, wrap="word", font=("Courier", 12))
    text_area.pack(fill="both", expand=True)
    text_area.insert("end", "🔍 Starting OCR processing...\n\n")

    # Queue for communication between threads
    result_queue = Queue()

    def worker():
        """Background worker to process OCR"""
        file_paths = []
        for root_dir, dirs, files in os.walk(selected_image_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    file_paths.append(os.path.join(root_dir, filename))

        for path in file_paths:
            try:
                # Detect printed or handwritten
                if models.predict_HvP(path) == 1:
                    raw_text = models.run_trOCR(models.printed_model, models.printed_processor, path)
                    category = "printed"
                else:
                    raw_text = models.run_trOCR(models.handwritten_model, models.handwritten_processor, path)
                    category = "handwritten"

                # Fix casing and punctuation
                fixed_text = models.restore_case_and_punctuation(raw_text)
                final_text = f"[{category}] {fixed_text}"
                
                # Put into queue for UI update
                result_queue.put(final_text)

            except Exception as e:
                result_queue.put(f"⚠ Error processing {os.path.basename(path)}: {e}")

        # Mark done
        result_queue.put(None)

    def poll_queue():
        """Check queue for new results and update UI"""
        try:
            while True:
                item = result_queue.get_nowait()
                if item is None:
                    text_area.insert("end", "\n✅ OCR processing complete.\n")
                    text_area.see("end")
                    return
                else:
                    text_area.insert("end", item + "\n\n")
                    text_area.see("end")
        except:
            pass
        ocr_window.after(200, poll_queue)

    # Start background thread
    threading.Thread(target=worker, daemon=True).start()

    # Start polling queue
    poll_queue()

    

def process_ocr_worker(file_paths):
    """Background thread worker: process OCR and put results in queue."""
    for path in file_paths:
        if models.predict_HvP(path) == 1:
            raw_text = models.run_trOCR(models.printed_model, models.printed_processor, path)
            category = "printed"
        else:
            raw_text = models.run_trOCR(models.handwritten_model, models.handwritten_processor, path)
            category = "handwritten"
        fixed_text = models.restore_case_and_punctuation(raw_text)
        final_text = f"({category}) {fixed_text}"


def process_image_with_line_segmentation():
    global selected_image_path
    path_label.config(text=f"Path: {selected_image_path}")
    if not selected_image_path:
        status_label.config(text="⚠ Please select a directory")
        return

    status_label.config(text="Running Line Segmentation...")
    root.update_idletasks()

    models.process_all_images()

    status_label.config(text="Line Segmentation Complete")
    root.update_idletasks()

def process_image_with_yolo():
    global selected_image_path
    if not selected_image_path:
        status_label.config(text="⚠ Please select an image first.")
        return

    status_label.config(text="Running YOLO detection...")
    root.update_idletasks()

    models.seperate_handwritten_printed_using_yolo(selected_image_path, yolo_output_dir)
    status_label.config(text="✅ YOLO detection complete!")
    selected_image_path = showing_yolo_results_dir
    path_label.config(text=f"Path: {selected_image_path}")

def select_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        selected_image_path = file_path
        path_label.config(text=f"Path: {selected_image_path}")

        img = Image.open(file_path)
        img.thumbnail((500, 500), Image.LANCZOS)
        img = add_black_border(img)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        status_label.config(text="✅ Image loaded. Ready for YOLO.")

def show_results():
    """Display images from YOLO output directory in a new scrollable window."""
    global yolo_images
    global selected_image_path
    yolo_images = []  # Clear any previous images

    if not os.path.exists(selected_image_path):
        status_label.config(text=f"⚠ output directory not found: {showing_yolo_results_dir}")
        return

    # Collect image files from YOLO output directory
    file_paths = []
    for root_dir, dirs, files in os.walk(selected_image_path):
        file_paths.extend(
            os.path.join(root_dir, filename)
            for filename in files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        )

    if not file_paths:
        status_label.config(text="⚠ No images found in output directory.")
        return

    # Create new window for YOLO results
    yolo_window = Toplevel(root)
    yolo_window.title("Results")
    yolo_window.geometry("800x600")

    # Set up scrollable canvas
    canvas = tk.Canvas(yolo_window, bg="white")
    scrollbar = tk.Scrollbar(yolo_window, orient="vertical", command=canvas.yview)
    grid_frame = tk.Frame(canvas, bg="white")

    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas_frame = canvas.create_window((0, 0), window=grid_frame, anchor="nw")

    def configure_canvas(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(canvas_frame, width=yolo_window.winfo_width())
    grid_frame.bind("<Configure>", configure_canvas)

    # Display images in a grid
    cols = 1
    for i, path in enumerate(file_paths):
        print(f'Image path: {path}')
        try:
            img = Image.open(path)
            img.thumbnail((400, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(grid_frame, image=img_tk, bg="white")
            img_label.grid(row=i // cols * 2, column=i % cols, padx=5, pady=5)
            yolo_images.append(img_tk)
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            status_label.config(text=f"⚠ Failed to load image {path}: {e}")
            continue

    # Force update to ensure canvas is properly sized
    canvas.update_idletasks()
    configure_canvas(None)
    status_label.config(text="✅ YOLO results displayed.")

def select_folder():
    """Opens a directory chooser dialog and updates the label with the selected path."""
    # Open the directory chooser dialog
    global selected_image_path
    folder_path = filedialog.askdirectory()

    selected_image_path = folder_path
    
    # Check if a folder was actually selected
    if folder_path:
        # Update the label's text with the selected path
        path_label.config(text=f"Selected Folder: {selected_image_path}")

# Tkinter UI Setup
root = tk.Tk()
root.title("Thesis")
root.geometry("700x750")

status_label = tk.Label(root, text="Initializing...", fg="blue")
status_label.pack()

# Button to open the folder dialog
select_button = tk.Button(root, text="Select Folder", command=select_folder)
select_button.pack()

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

yolo_btn = tk.Button(root, text="Run YOLO", command=process_image_with_yolo)
yolo_btn.pack()

line_segment_btn = tk.Button(root, text="Run line segmentation", command=process_image_with_line_segmentation)
line_segment_btn.pack()

OCR_btn = tk.Button(root, text="Apply OCR", command=generate_ocr_result)
OCR_btn.pack()

show_results_btn = tk.Button(root, text="Show Results", command=show_results)
show_results_btn.pack()

refresh_btn = tk.Button(root, text="Refresh", command=refresh_ui)
refresh_btn.pack()

path_label = tk.Label(root, text="No image selected", wraplength=600, justify="center")
path_label.pack()

image_label = tk.Label(root, bg="white", width=500, height=500)
image_label.pack()

yolo_output_image = tk.Label(root, bg="white", width=500, height=500)
yolo_output_image.pack()

# Load models in background thread
threading.Thread(target=load_models_background, daemon=True).start()

root.mainloop()