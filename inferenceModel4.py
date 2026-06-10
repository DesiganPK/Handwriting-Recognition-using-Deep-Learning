'''import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress INFO, WARNING, and ERROR logs
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
warnings.simplefilter("ignore", category=DeprecationWarning)'''
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from transformers.utils.logging import set_verbosity_error
import google.generativeai as genai
genai.configure(api_key="")#Enter your API key here
set_verbosity_error()
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import shutil

folder_path = os.path.join(os.getcwd(),"segmented")  # Replace with the actual path to your folder

def clear_folder(folder_path):
    """Deletes all files and subdirectories within the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def output():
    # Load processor
    processor = TrOCRProcessor.from_pretrained("./localModel")

    # Load model - explicitly specifying the weight file
    model = VisionEncoderDecoderModel.from_pretrained(
        "./localModel",
        ignore_mismatched_sizes=True  # This allows loading even if some weights are missing
    )

    i = 0
    text = ""
    while True:
        file_path = os.path.join(os.getcwd(), "segmented", f"segment{i}.png")
        if not os.path.exists(file_path):
            break
        image = Image.open(file_path).convert("RGB")
        if image is None:
            break
        else:
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():  # No gradient needed for inference
                generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            prediction_text = generated_text
            text += prediction_text + " "
            i += 1
    
    textReturn = text
            
    model_ai = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model_ai.generate_content(f"Rectify the spelling mistakes. Input Sentence:{text}.Return the complete sentence")
    print("Full predicted text = " + response.text)
    text = response.text

    # Clear the segmented folder
    clear_folder(folder_path)

    # Dynamically find and clear the YOLO 'predict' folder
    detect_path = os.path.join(os.getcwd(), "runs", "detect")
    if os.path.exists(detect_path):
        predict_folders = [f for f in os.listdir(detect_path) if f.startswith("predict")]
        for folder in predict_folders:
            clear_folder(os.path.join(detect_path, folder))

    return [textReturn, text]

if __name__ == "__main__":
    output()
