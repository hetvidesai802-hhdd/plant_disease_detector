# LeafCare — Plant Disease Detector

A beginner-friendly end-to-end image-classification project:
- PlantVillage leaf image dataset (38 crop/disease classes)
- ImageNet-pretrained ResNet18 fine-tuned with PyTorch
- Flask upload interface with predicted class and confidence score

## 1. Requirements
Python 3.10 or 3.11 is recommended. Training is much faster with an NVIDIA GPU, but CPU training is supported (it can take considerably longer).

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with your installed Python version.

## 3. Download and prepare the dataset
```powershell
python prepare_data.py
```
This downloads the color configuration of PlantVillage from Hugging Face and creates:
```text
data/
  train/<class_name>/
  val/<class_name>/
  test/<class_name>/
  class_names.json
```
The official train/test split is retained; 10% of the official training split is held out for validation.

## 4. Train the model
```powershell
python train.py
```
The script uses ImageNet-pretrained ResNet18, replaces its final layer for the 38 PlantVillage classes, and fine-tunes layer4 plus the classifier. It prints validation accuracy each epoch and final held-out test accuracy. The trained checkpoint is saved to:
`models/plant_disease_resnet18.pth`

The first run may download pretrained ResNet18 weights. If training is too slow on CPU, reduce `EPOCHS` or `BATCH_SIZE` in `train.py`.

## 5. Start the Flask app
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser. Upload a clear JPG, PNG, or WEBP leaf image.

## Project structure
```text
plant_disease_detector/
├── app.py
├── train.py
├── prepare_data.py
├── requirements.txt
├── models/                 # trained model checkpoint appears here
├── data/                   # downloaded/prepared dataset (not in zip)
├── templates/index.html
└── static/style.css
```

## Important limitations
PlantVillage images were largely captured in controlled conditions with relatively plain backgrounds. A high benchmark score does not guarantee equivalent performance on real farm photos. The model only predicts among its 38 known classes; it does not reliably detect unknown plants, pests, nutrient deficiencies, or diseases absent from the dataset. Confidence is the model's softmax score, not a calibrated probability. Treat results as preliminary guidance and confirm with an agricultural expert before taking action.

## Dataset and method references
- PlantVillage dataset: Mohanty, Hughes & Salathé (2016), *Using Deep Learning for Image-Based Plant Disease Detection*.
- Dataset card: https://huggingface.co/datasets/mohanty/PlantVillage
- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
