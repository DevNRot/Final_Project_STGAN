
# 🎨 Final Project DLIP — STGAN

This repository contains the code, models, and resources for a Conditional GAN (CGAN) project, developed as part of the Kaggle challenge *I'm Something of a Painter Myself*, for the Deep Learning Image Processing (DLIP) course.

---

## 📂 Project Structure

- `data/` — Folder containing the dataset images used for training
  - `monet_jpg/` — folder to contain Monet images 
  - `photo_jpg/` — folder to contain training photographs
- `STGAN/` — Main project code: model definitions, custom layers, utility functions
  - `models_STGAN.py` — Contains the generator and discriminator model definitions
  - `utils.py` — Utility functions for visualizing results
  - `Data_loader_STGAN.py` — custom Dataloader objects
  - `custom_layers_STGAN.py` — custom layers for the models in the cGAN
  - `checkpoints/` — Directory for saving the current model checkpoints
  - `Samples/` — Generated images, graphs, and models from previous training runs
  - `Final_Project_Training_Loop_v3.ipynb` — Training notebook for the CGAN
- `inference_and_metric.ipynb` — Notebook for generating new images using the trained model and computing the MiFID score for evaluation

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/DevNRot/final_project_STGAN.git
   cd final-project-dlip```
   
2. Place the dataset is placed inside the `data/` folder, where the Monet images are in the nested monet_jpg 
   folder and the photographs are in the nested photos_jpg folder.

3. Run the training notebook:
   - `Final_Project_Training_Loop_v3.ipynb`

4. Use the inference notebook to generate new images and compute the MiFID score:
   - `inference_and_metric.ipynb`

---

## 📊 Model Summary

We implemented a Conditional GAN (CGAN) architecture for image generation, including:
- Custom generator and discriminator architectures in `STGAN/models_STGAN.py`
- Data loading, training loop, and loss functions
- Image generation and evaluation pipeline using MiFID score

---

## 📝 Notes

- All code is in Python 3.x
- Dependencies are listed in the first cell of each notebook

---

## 📣 Credits

Developed by Debbie Rotman and Carmit Kaye for the BGU MSc DLIP course, Spring 2025.
