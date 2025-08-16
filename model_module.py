import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import cv2
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import binary_closing
import secrets      # cryptographically‑secure RNG
import base64       # for compact ASCII/“number + letter” output
import uuid
from ultralytics import YOLO
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from HvP import SimpleCNN  # Import the model from the other file
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
from line_segmentation import calculate_projection_profile_and_crop_lines_with_lines
import pkg_resources
from symspellpy import SymSpell
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pkg_resources
from symspellpy import SymSpell
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hand_written_model_id = "microsoft/trocr-large-handwritten"
printed_model_id = "microsoft/trocr-base-printed"
from transformers import T5ForConditionalGeneration, T5Tokenizer
yolo_output_dir = r"kaggle/output/"
base_input_dir = r"kaggle/output/cropped_outputs"
base_output_printed = r"kaggle/output/cropped_outputs_line"
base_graph_folder = r"kaggle/output/line_graphs"
input_image_path = ""

class Models:
    def __init__(self):
        self.yolo_model = YOLO("kaggle/input/weights/last_100.pt")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
        self.nlp_model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
        self.punct_pipeline = pipeline("token-classification", model=self.nlp_model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        self.hvp_model = SimpleCNN().to(DEVICE)
        self.hvp_model.load_state_dict(torch.load("kaggle/input/weights/final_model_weights_HvP.pth", map_location=DEVICE))
        self.printed_processor = TrOCRProcessor.from_pretrained(printed_model_id)
        self.printed_model = VisionEncoderDecoderModel.from_pretrained(printed_model_id).to(DEVICE)
        self.handwritten_processor = TrOCRProcessor.from_pretrained(hand_written_model_id)
        self.handwritten_model = VisionEncoderDecoderModel.from_pretrained(hand_written_model_id).to(DEVICE)
            # Test transforms
        self.test_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


    def seperate_handwritten_printed_using_yolo(self, image_path, folder_name):
        results = self.yolo_model([image_path])  
        image = cv2.imread(image_path)
        base_name = os.path.basename(image_path)

        base_crop_folder = "cropped_outputs"
        base_graph_folder = "graph_outputs"

        # Class index to folder name
        class_to_folder = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            9: "Text",
            10: "Title"
        }

        for result in results:
            boxes = result.boxes
            cls = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            # Iterate over detections
            for i, box in enumerate(xyxy):
                class_id = int(cls[i])
                if class_id in class_to_folder:  # Only process selected classes
                    x1, y1, x2, y2 = map(int, box)
                    cropped = image[y1:y2, x1:x2]

                    # Save crop
                    crop_folder = os.path.join(folder_name, base_crop_folder, class_to_folder[class_id])
                    os.makedirs(crop_folder, exist_ok=True)
                    filename = f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(os.path.join(crop_folder, filename), cropped)

            # Save graph image
            graph = result.plot()
            graph_folder = os.path.join(folder_name, base_graph_folder)
            os.makedirs(graph_folder, exist_ok=True)
            cv2.imwrite(os.path.join(graph_folder, base_name), graph)

    def process_all_images(self):
        for root, dirs, files in os.walk(base_input_dir):
            image_files = sorted(
                [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            )
            for file in image_files:
                image_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                try:
                    calculate_projection_profile_and_crop_lines_with_lines(image_path, folder_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    
    # --- Load and preprocess the image ---
    def predict_HvP(self,image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.test_transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
        
        # --- Make prediction ---
        # model.eval()
        with torch.no_grad():
            output = self.hvp_model(image_tensor)
            predicted_class = output.argmax(1).item()
        
        # --- Map class index to class name ---
        class_names = ["handwritten", "printed"]  # Get class names from dataset
        print(f"Predicted class label: {class_names[predicted_class]}")
        return predicted_class

        # OCR runner
    def run_trOCR(self, model, processor, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
        generated_ids = model.generate(pixel_values, max_new_tokens=1000)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(f"{os.path.basename(image_path)} -> {generated_text}")
        return generated_text

    def restore_case_and_punctuation(self,text: str):
        # Step 1: Spell Correction
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        corrected_text = suggestions[0].term if suggestions else text
        
        # Step 2: Case Restoration (simple heuristic)
        corrected_text = corrected_text.strip()
        if corrected_text:  
            corrected_text = corrected_text[0].upper() + corrected_text[1:]
        corrected_text = corrected_text.replace(" i ", " I ")
        
        # Step 3: Punctuation Restoration
        tokens = corrected_text.split()
        predictions = self.punct_pipeline(corrected_text)
        
        restored_text = ""
        for i, token in enumerate(tokens):
            restored_text += token
            # Add punctuation if predicted
            if i < len(predictions) and predictions[i]['word'] in [".", ",", "?", "!"]:
                restored_text += predictions[i]['word']
            restored_text += " "
        
        return restored_text.strip()
    
    def OCR_output(self):
        folder_dir = "kaggle/output/cropped_outputs_line/cropped_outputs/"
        for paths in os.listdir(folder_dir):
            
            input_dir = os.path.join(folder_dir,paths)       
            # Collect image file paths
            file_paths = [
                os.path.join(input_dir, filename)
                for filename in (os.listdir(input_dir))
            ]

            # Loop through and run OCR
            for path in file_paths:
                # img = cv2.imread(path)
                # plt.imshow(img, cmap='gray')
                raw_text = ""
                if self.predict_HvP(path) == 1:
                    raw_text = ((self.run_trOCR(self.printed_model, self.printed_processor, path)))
                else:
                    raw_text = ((self.run_trOCR(self.handwritten_model, self.handwritten_processor, path)))
                
                print("Raw Text:", raw_text)
                fixed_text = self.restore_case_and_punctuation(raw_text)
                print("Fixed Text:", fixed_text)