# Handwriting Recognition using Deep Learning

## Overview

Handwriting Recognition using Deep Learning is an end-to-end handwritten document digitization system that combines object detection, transformer-based OCR, and Generative AI to convert handwritten images and PDF documents into structured digital text.

The system uses **YOLOv11** to detect and segment individual handwritten words from a document. Each detected word is then passed through a **fine-tuned TrOCR (Transformer OCR)** model trained on the **IAM Words Dataset** for accurate handwritten text recognition.

To further improve readability and usability, the extracted text is processed using the **Google Gemini API**, which automatically corrects spelling mistakes, improves formatting, and reconstructs cleaner text output.

The project also supports **multi-page PDF documents**, automatically converting PDF pages into images and processing each page through the complete OCR pipeline.

---

## Key Highlights

* End-to-End Handwriting Recognition System
* YOLOv11-based Word Detection
* Fine-Tuned TrOCR Word Recognition Model
* Character Error Rate (CER): **4%**
* Gemini-powered Text Refinement
* Image-to-Text Conversion
* PDF-to-Text Conversion
* Multi-page Document Processing
* Interactive Gradio Web Interface

---

## Features

### Handwritten Text Recognition

* Detects handwritten words using YOLOv11
* Recognizes handwritten text using a fine-tuned TrOCR model
* Supports multi-line handwritten documents
* Reconstructs complete text from detected words

### AI-Powered Text Enhancement

* Corrects spelling mistakes
* Improves punctuation
* Fixes formatting issues
* Produces cleaner and more readable output

### PDF Processing

* Converts PDF pages into images
* Processes each page independently
* Merges extracted text across pages
* Supports multi-page handwritten documents

### User Interface

* Simple Gradio-based web application
* Upload images directly through the browser
* View both raw OCR output and AI-enhanced output

---

## System Architecture

```text
                  Input Document
                (Image or PDF)
                         │
                         ▼
               PDF Page Extraction
                  (if PDF input)
                         │
                         ▼
                    Page Images
                         │
                         ▼
                  YOLOv11 Detector
          (Word Detection & Segmentation)
                         │
                         ▼
                   Word Cropping
                         │
                         ▼
              Fine-Tuned TrOCR Model
               (Word Recognition)
                         │
                         ▼
               Reconstructed Text
                         │
                         ▼
                  Gemini API
     (Spelling & Formatting Correction)
                         │
                         ▼
                    Final Output
```

---

## Technology Stack

### Deep Learning

* YOLOv11
* Fine-Tuned TrOCR
* PyTorch
* Hugging Face Transformers

### Computer Vision

* OpenCV
* NumPy

### Generative AI

* Google Gemini API

### Frontend

* Gradio

### Programming Language

* Python

---

## Dataset

### IAM Words Dataset

The OCR model was fine-tuned using the IAM Words Dataset, a widely used benchmark dataset for handwritten text recognition research.

The dataset contains:

* Thousands of handwritten word images
* Multiple handwriting styles
* Ground-truth transcriptions
* Real-world handwritten samples

---

## Model Performance

### Fine-Tuned TrOCR

| Metric                     | Value |
| -------------------------- | ----- |
| Character Error Rate (CER) | 4%    |

The fine-tuned TrOCR model achieved a Character Error Rate (CER) of approximately **4%**, demonstrating strong handwritten word recognition performance.

---

## Workflow

### Step 1: Input Upload

The user uploads either:

* A handwritten image
* A handwritten PDF document

### Step 2: PDF Processing (Optional)

If a PDF is uploaded:

* Each page is converted into an image
* Pages are processed individually

### Step 3: Word Detection

YOLOv11 detects handwritten words and generates bounding boxes around them.

### Step 4: Word Extraction

Detected words are cropped from the original document image.

### Step 5: Word Recognition

Each cropped word is passed through the fine-tuned TrOCR model to generate text predictions.

### Step 6: Text Reconstruction

Recognized words are arranged according to their positions to reconstruct the original text.

### Step 7: AI-Based Enhancement

The reconstructed text is sent to Gemini API, which:

* Corrects spelling mistakes
* Improves grammar
* Enhances formatting
* Produces cleaner final text

---

## Project Structure

```text
HandWritingRecognition/
│
├── app.py
├── app_pdf.py
├── predict.py
├── inferenceModel4.py
├── words.py
├── checkout.py
├── requirements.txt
├── README.md
│
├── localModel/
│   ├── config.json
│   ├── tokenizer files
│   └── model.safetensors
│
├── runs/
├── outputs/
└── assets/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/DesiganPK/Handwriting-Recognition-using-Deep-Learning.git

cd Handwriting-Recognition-using-Deep-Learning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Setup

Due to GitHub file size limitations, trained model weights are not included in the repository.

Download the required model files and place them in the appropriate directories:

### YOLOv11 Model

```text
yolo_custom.pt
```

### Fine-Tuned TrOCR Model

```text
localModel/
```

Update the paths in the project files if necessary.

---

## Running the Application

### Image-to-Text Recognition

```bash
python app.py
```

### PDF-to-Text Recognition

```bash
python app_pdf.py
```

After launching, open the Gradio URL displayed in the terminal.

---

