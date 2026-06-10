# app_pdf.py
import gradio as gr
import os
import shutil
import time

# Import your processing functions
from pdf_utils import convert_pdf_to_images # Handles PDF conversion
from predict import input         # Handles YOLO segmentation (was 'input')
from inferenceModel4 import output          # Handles text inference/GenAI (ensure this exists)

# --- Configuration ---
SEGMENTED_FOLDER = "segmented" # Folder where input_processor saves crops

# --- Helper Function ---
def cleanup_directory(dir_path):
    """Removes a directory if it exists."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Cleaned up directory: {dir_path}")
        except OSError as e:
            print(f"Error removing directory {dir_path}: {e}")

def cleanup_prediction_runs():
    """Removes the default YOLO prediction run folders."""
    predict_dir_base = "runs/detect"
    if os.path.exists(predict_dir_base):
        # List subdirectories like 'predict', 'predict2', etc.
        for item in os.listdir(predict_dir_base):
            item_path = os.path.join(predict_dir_base, item)
            if os.path.isdir(item_path) and item.startswith("predict"):
                 cleanup_directory(item_path)


# --- Main Processing Function for Gradio ---
def process_pdf(pdf_file):
    """
    Processes an uploaded PDF file: converts to images, runs prediction
    and inference on each page, and combines the results.

    Args:
        pdf_file (FileStorage or similar object from Gradio): The uploaded PDF file.

    Returns:
        tuple: (combined_predicted_text, combined_genai_text)
    """
    if pdf_file is None:
        return "Error: No PDF file uploaded.", "Error: No PDF file uploaded."

    pdf_path = pdf_file.name # Get the temporary path where Gradio stores the upload
    print(f"Received PDF for processing: {pdf_path}")

    temp_image_dir = None
    all_predicted_text = []
    all_genai_text = []

    try:
        # 1. Convert PDF to Images
        start_time = time.time()
        image_paths, temp_image_dir = convert_pdf_to_images(pdf_path)
        if not image_paths:
            # Error handled and logged within convert_pdf_to_images
            return "Error: PDF conversion failed.", "Error: PDF conversion failed."
        conversion_time = time.time() - start_time
        print(f"PDF conversion took {conversion_time:.2f} seconds.")

        # 2. Process Each Image Page
        total_pages = len(image_paths)
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            print(f"\n--- Processing Page {page_num}/{total_pages} ({os.path.basename(image_path)}) ---")
            page_start_time = time.time()

            # Clean the segmentation folder before processing the next page
            cleanup_directory(SEGMENTED_FOLDER)
            # Optional: Clean up previous YOLO runs if they interfere
            # cleanup_prediction_runs() # Be cautious if other processes use these

            # Run YOLO word segmentation
            try:
                input(image_path) # Saves segments to SEGMENTED_FOLDER
                print(f"Page {page_num}: Segmentation complete.")
            except Exception as e:
                print(f"Error during segmentation for page {page_num}: {e}")
                all_predicted_text.append(f"--- Page {page_num}: Segmentation Error ---")
                all_genai_text.append(f"--- Page {page_num}: Segmentation Error ---")
                continue # Skip inference if segmentation failed

            # Run inference/GenAI on the segmented words
            # Ensure output() reads from SEGMENTED_FOLDER
            if not os.path.exists(SEGMENTED_FOLDER) or not os.listdir(SEGMENTED_FOLDER):
                 print(f"Page {page_num}: No segments found in '{SEGMENTED_FOLDER}'. Skipping inference.")
                 all_predicted_text.append(f"--- Page {page_num}: No text detected ---")
                 all_genai_text.append(f"--- Page {page_num}: No text detected ---")
                 continue

            try:
                text_return, text_genai = output() # Assumes output() reads from SEGMENTED_FOLDER
                all_predicted_text.append(f"--- Page {page_num} ---\n{text_return}")
                all_genai_text.append(f"--- Page {page_num} ---\n{text_genai}")
                print(f"Page {page_num}: Inference complete.")
            except Exception as e:
                print(f"Error during inference for page {page_num}: {e}")
                all_predicted_text.append(f"--- Page {page_num}: Inference Error ---")
                all_genai_text.append(f"--- Page {page_num}: Inference Error ---")

            page_time = time.time() - page_start_time
            print(f"Page {page_num} processing took {page_time:.2f} seconds.")

        # 3. Combine Results
        combined_predicted = "\n\n".join(all_predicted_text)
        combined_genai = "\n\n".join(all_genai_text)

        total_time = time.time() - start_time
        print(f"\n--- Total Processing Time: {total_time:.2f} seconds ---")

        return combined_predicted, combined_genai

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Log the full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        return f"Error: An unexpected error occurred during processing: {e}", f"Error: An unexpected error occurred during processing: {e}"

    finally:
        # 4. Cleanup
        print("Starting cleanup...")
        # Remove temporary image directory
        if temp_image_dir:
            cleanup_directory(temp_image_dir)
        # Remove the segmentation folder from the last page
        cleanup_directory(SEGMENTED_FOLDER)
        # Optional: Clean up YOLO runs
        # cleanup_prediction_runs()


# --- Gradio Interface Definition ---
with gr.Blocks() as demo:
    gr.Markdown("# PDF Text Extraction and Analysis")
    gr.Markdown("Upload a PDF file. The system will extract text from each page using YOLO word detection and then run further analysis.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

    with gr.Row():
        submit_button = gr.Button("Process PDF")

    with gr.Row():
        output_predicted = gr.Textbox(label="Predicted Text (All Pages)", lines=15)
        output_genai = gr.Textbox(label="GenAI Text (All Pages)", lines=15)

    submit_button.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[output_predicted, output_genai]
    )

# --- Launch the App ---
if __name__ == "__main__":
    # Ensure the script that calls output() can find necessary models/data
    # Make sure inferenceModel4.py and predict.py are in the same directory
    # or accessible via Python's path.
    demo.launch(share=True) # Set share=False if you don't need public access