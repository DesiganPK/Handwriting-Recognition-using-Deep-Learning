import os
import cv2
import numpy as np
import google.generativeai as genai
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import shutil

# Configure Generative AI
genai.configure(api_key="")#Enter your API key here

# Folder paths
segmented_folder = os.path.join(os.getcwd(), "segmented")

def clear_folder(folder_path):
    """Deletes all files and subdirectories within the specified folder."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def output():
    # Load model configurations
    from mltu.configs import BaseModelConfigs
    configs = BaseModelConfigs.load("202301111911/configs.yaml")

    # Initialize the model
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Process segmented images
    i = 0
    text = ""
    while True:
        file_path = os.path.join(segmented_folder, f"segment{i}.png")
        if not os.path.exists(file_path):
            break
        image = cv2.imread(file_path)
        if image is None:
            break
        else:
            prediction_text = model.predict(image)
            text += prediction_text + " "
            i += 1

    # Use Generative AI to refine the text
    model_ai = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model_ai.generate_content(
        f"Rectify the spelling mistakes. Input Sentence: {text}. Return the complete sentence."
    )
    print("Full predicted text = " + response.text)
    text = response.text

    # Clear the segmented folder
    clear_folder(segmented_folder)

    return text

if __name__ == "__main__":
    output()
