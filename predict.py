from ultralytics import YOLO
import cv2
import os
import numpy as np # Import numpy for calculating averages

def input(image_path): # Renamed function for clarity
    """
    Detects words in an image using YOLO, sorts them into reading order
    (handling tilted lines), and saves cropped word images.

    Args:
        image_path (str): Path to the input image file.
    """
    # --- YOLO Detection ---
    # Load YOLO model
    model = YOLO("yolo_custom.pt") # Ensure this model path is correct

    # Define the base directory for prediction outputs to keep it consistent
    predict_dir = "runs/detect/predict" # Default Ultralytics output dir base

    # Run YOLO prediction on the input image
    # Explicitly set project and name to control output directory
    # This helps reliably find the label file later.
    results = model.predict(source=image_path,
                            conf=0.4,
                            line_width=1,
                            save_crop=False,
                            save_txt=True,
                            project="runs/detect", # Base directory
                            name="predict",       # Subdirectory (will create predict, predict2, etc. if exists)
                            exist_ok=True)        # Overwrite if 'predict' exists

    # --- Result Parsing ---
    # Load the input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    h, w, _ = img.shape

    # Determine the exact path to the YOLO output label file
    # Ultralytics might create predict, predict2, predict3...
    # We assume the latest one is the one just created.
    # If results object provides the save dir, use that. Otherwise, find the latest.
    save_dir = results[0].save_dir # Get the actual save directory from results
    txt_path = os.path.join(save_dir, 'labels', os.path.splitext(os.path.basename(image_path))[0] + '.txt')

    # Check if the YOLO output file exists
    if not os.path.exists(txt_path):
        print(f"YOLO output label file not found at: {txt_path}")
        print("Please ensure YOLO prediction ran successfully and saved the text file.")
        return

    # Parse YOLO output and convert to absolute coordinates including y_center
    boxes = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Warning: YOLO output file is empty: {txt_path}")
                # Decide if processing should stop or continue with no boxes
                # return # Option: stop if no boxes found

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Warning: Skipping malformed line in {txt_path}: {line.strip()}")
                    continue
                _, x_center_rel, y_center_rel, box_w_rel, box_h_rel = map(float, parts)

                # Calculate absolute coordinates
                x_center = x_center_rel * w
                y_center = y_center_rel * h
                box_w = box_w_rel * w
                box_h = box_h_rel * h

                # Calculate corner coordinates (ensure they are within image bounds)
                x1 = max(0, int(x_center - box_w / 2))
                y1 = max(0, int(y_center - box_h / 2))
                x2 = min(w, int(x_center + box_w / 2))
                y2 = min(h, int(y_center + box_h / 2))

                # Store box details including center y and height for sorting
                if box_w > 0 and box_h > 0: # Ensure valid box dimensions
                    boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'yc': y_center, 'h': box_h})
    except Exception as e:
        print(f"Error reading or parsing YOLO output file {txt_path}: {e}")
        return

    if not boxes:
        print("No bounding boxes were extracted from the YOLO output.")
        return # Stop if no boxes


    # --- Improved Sorting Logic ---
    def sort_boxes_robust(boxes, y_center_tolerance_ratio=0.5):
        """
        Sorts boxes into lines robustly, handling tilted text.

        Args:
            boxes (list): List of box dictionaries [{'x1','y1','x2','y2','yc','h'}, ...].
            y_center_tolerance_ratio (float): Tolerance for grouping boxes into the same line,
                                             relative to the box's height. E.g., 0.5 means
                                             the box center must be within half its height
                                             of the line's average center.

        Returns:
            list: List of sorted box tuples [(y1, x1, x2, y2), ...].
        """
        if not boxes:
            return []

        # Sort primarily by average y_center, secondarily by x1
        # This gives a top-to-bottom, left-to-right initial order
        boxes.sort(key=lambda b: (b['yc'], b['x1']))

        sorted_lines = []
        current_line = []

        for box in boxes:
            if not current_line:
                # Start the first line
                current_line.append(box)
            else:
                # Calculate the average y_center of the boxes currently in the line
                avg_yc_line = np.mean([b['yc'] for b in current_line])
                # Calculate the tolerance based on the current box's height
                # This allows taller boxes to have a larger vertical deviation tolerance
                tolerance = box['h'] * y_center_tolerance_ratio

                # Check if the current box's y_center is close enough to the line's average center
                if abs(box['yc'] - avg_yc_line) < tolerance:
                    # Add box to the current line
                    current_line.append(box)
                else:
                    # Finish the current line: sort it by x1
                    current_line.sort(key=lambda b: b['x1'])
                    sorted_lines.extend(current_line)
                    # Start a new line with the current box
                    current_line = [box]

        # Add the last processed line
        if current_line:
            current_line.sort(key=lambda b: b['x1'])
            sorted_lines.extend(current_line)

        # Convert back to the original (y1, x1, x2, y2) tuple format for cropping
        final_boxes = [(b['y1'], b['x1'], b['x2'], b['y2']) for b in sorted_lines]
        return final_boxes

    # Sort the boxes using the robust function
    # You might need to tune the 'y_center_tolerance_ratio' (0.3 - 0.7 is a common range)
    sorted_boxes = sort_boxes_robust(boxes, y_center_tolerance_ratio=0.5)

    # --- Cropping and Saving ---
    # Create output directory for segmented images
    output_dir = os.path.join(os.getcwd(), "segmented")
    os.makedirs(output_dir, exist_ok=True)

    # Crop and save segmented images in the sorted order
    print(f"Saving {len(sorted_boxes)} segmented word images to {output_dir}...")
    for i, (y1, x1, x2, y2) in enumerate(sorted_boxes):
        # Ensure coordinates are valid integers for cropping
        y1, x1, x2, y2 = int(y1), int(x1), int(x2), int(y2)
        # Crop the image using NumPy slicing [y1:y2, x1:x2]
        crop = img[y1:y2, x1:x2]

        # Check if the crop is valid before saving
        if crop.size == 0:
            print(f"Warning: Skipping empty crop for box {i} with coords ({y1},{x1},{x2},{y2})")
            continue

        # Construct filename (consider adding padding for better alphanumeric sorting if needed)
        segment_filename = f"segment{i}.png" # e.g., segment_0000.png, segment_0001.png
        segment_path = os.path.join(output_dir, segment_filename)

        try:
            cv2.imwrite(segment_path, crop)
        except Exception as e:
            print(f"Error saving segment {segment_filename}: {e}")

    print("Finished processing.")


# --- Main Execution ---
if __name__ == "__main__":
    image_to_process = "page_5.jpg" # Use the uploaded image name
    if not os.path.exists(image_to_process):
        print(f"Error: Input image '{image_to_process}' not found in the current directory.")
    else:
        input_processor(image_to_process)
    # call(["python", "inferenceModel4.py"]) # Uncomment if needed