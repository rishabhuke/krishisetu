# 🌿 KrishiSetu — AI-Powered Crop Disease Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-94%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Design and Development of a Machine Learning-Based Model for Crop Disease Detection with a Web-Based Interface**
>
> B.Tech Project — Madhav Institute of Technology and Science, Gwalior

---

## 🎯 Problem Statement

India loses approximately **₹50,000 crore annually** due to crop diseases. Over **70% of farmers** lack access to early disease detection tools, leading to significant yield losses. Traditional diagnosis requires agricultural experts which are often unavailable in rural areas.

**KrishiSetu bridges this gap** by providing instant AI-powered crop disease detection accessible to every farmer through a simple web interface.

---

## 🌟 Live Demo

> 🔗 **[Try KrishiSetu Live](https://huggingface.co/spaces/rishabhuke/krishisetu)** *(Coming Soon)*

---

## 📸 Screenshots

### Home Page
![Home Page](docs/screenshots/home.png)

### Disease Detection
![Detection](docs/screenshots/detect.png)

### Farm Shop
![Shop](docs/screenshots/shop.png)

### Mandi Prices
![Mandi](docs/screenshots/mandi.png)

### Weather Forecast
![Weather](docs/screenshots/weather.png)

### Govt Schemes
![Schemes](docs/screenshots/schemes.png)

### Crop Calendar
![Calendar](docs/screenshots/calendar.png)

### Scan History
![History](docs/screenshots/history.png)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **AI Disease Detection** | Upload leaf photo → instant diagnosis with 94% accuracy |
| 💊 **Treatment Guide** | Disease-specific remedies with severity assessment |
| 🛒 **Farm Shop** | Auto-recommended seeds, medicines and fertilizers |
| 🌤️ **Weather Forecast** | Real-time weather with farming advice |
| 💰 **Mandi Prices** | Live crop market rates across India |
| 📅 **Crop Calendar** | Sowing and harvesting guide by season |
| 📋 **Govt Schemes** | 12 government subsidies and schemes |
| 📊 **Scan History** | Track all past disease detections |
| 🔐 **Authentication** | Secure login/signup with SQLite database |

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
        ↓
MobileNetV2 (pretrained on ImageNet)
[154 layers, frozen base]
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, ReLU) + Dropout(0.4)
        ↓
Dense(128, ReLU) + Dropout(0.3)
        ↓
Dense(38, Softmax)
        ↓
Disease Prediction + Confidence Score
```

### Training Results

| Model | Validation Accuracy | Parameters | Size |
|---|---|---|---|
| CNN from Scratch | 93.4% | 26M | 299MB |
| MobileNetV2 (Phase 1) | 91.7% | 368K | 30MB |
| **MobileNetV2 (Fine-tuned)** | **94.02%** | **1.9M** | **30MB** |

---

## 📊 Dataset

- **Source:** PlantVillage Dataset
- **Total Images:** 54,305
- **Classes:** 38 (26 diseases + 12 healthy)
- **Crops:** 11 types (Tomato, Potato, Corn, Grape, Apple, Pepper, Orange, Peach, Strawberry, Cherry, Blueberry)
- **Split:** 80% Training (43,444) / 20% Validation (10,861)

---

## 🛠️ Tech Stack

### Machine Learning
- **TensorFlow 2.16** — Deep learning framework
- **MobileNetV2** — Transfer learning backbone
- **OpenCV** — Image preprocessing
- **NumPy / Pandas** — Data manipulation
- **Scikit-learn** — Model evaluation metrics

### Web Application
- **Flask** — Python web framework
- **SQLite** — User authentication database
- **HTML5 / CSS3** — Frontend
- **JavaScript** — Interactive UI
- **OpenWeatherMap API** — Weather data

### Tools & Deployment
- **Jupyter Notebook** — Model development
- **Git / GitHub** — Version control
- **Hugging Face Spaces** — Deployment

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Anaconda / Miniconda
- 4GB RAM minimum

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rishabhuke/krishisetu.git
cd krishisetu

# 2. Create conda environment
conda create -n cropenv python=3.10 -y
conda activate cropenv

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the trained model
# Place mobilenet_finetuned.keras in the model/ folder
# Download from Google Drive: [Add your link here]

# 5. Run the application
cd app
python app.py
```

### Open in browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
krishisetu/
│
├── app/                          # Flask web application
│   ├── app.py                    # Main application & routes
│   ├── static/
│   │   ├── css/style.css         # Complete stylesheet
│   │   └── js/cart.js            # Shopping cart logic
│   └── templates/                # HTML templates
│       ├── index.html            # Landing page
│       ├── detect.html           # Upload page
│       ├── result.html           # Results page
│       ├── shop.html             # Farm shop
│       ├── weather.html          # Weather forecast
│       ├── mandi.html            # Mandi prices
│       ├── calendar.html         # Crop calendar
│       ├── schemes.html          # Govt schemes
│       ├── history.html          # Scan history
│       ├── login.html            # Authentication
│       └── signup.html           # Registration
│
├── model/                        # Trained models
│   └── class_names.json          # 38 disease class labels
│   (mobilenet_finetuned.keras — download separately)
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── docs/screenshots/             # App screenshots
├── report/                       # Charts and visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📈 Model Performance

### Per-Class Accuracy
![Accuracy Chart](report/per_class_accuracy.png)

### Confusion Matrix
![Confusion Matrix](report/confusion_matrix.png)

### Training History
![Training](report/training_history_scratch.png)

```
Overall Accuracy:    94.02%
Average Confidence:  95.34%
Total Evaluated:     10,861 images
```

---

## 🔬 Key Technical Decisions

### Why MobileNetV2?
- **10x smaller** than CNN from scratch (30MB vs 299MB)
- **3x faster** training (2 hours vs 6 hours)
- **Higher accuracy** (94% vs 93.4%)
- **Mobile-ready** — can be deployed on edge devices

### Why Transfer Learning?
MobileNetV2 was pretrained on ImageNet (1.2M images, 1000 classes). It already knows how to detect edges, textures and shapes. We only needed to teach it crop-specific disease patterns — saving massive training time.

### Why Flask?
Lightweight, Python-native, easy to integrate with TensorFlow. Perfect for ML model serving without the overhead of Django.

---

## ⚠️ Limitations & Future Work

| Limitation | Future Solution |
|---|---|
| Domain gap (lab vs field photos) | Collect real-world training data |
| 38 fixed disease classes | Expand dataset with more crops |
| No mobile app | React Native / Flutter app |
| Static mandi prices | Integrate AGMARKNET live API |
| Single language UI | Hindi/Marathi/regional languages |

---

## 🎓 Academic Details

- **Project Title:** Design and Development of a Machine Learning-Based Model for Crop Disease Detection with a Web-Based Interface
- **Student:** Rishabh Uke
- **Degree:** B.Tech (Artificial Intelligence and Machine Learning)
- **Institute:** Madhav Institute of Technology and Science, Gwalior
- **Year:** 2024-25

---

## 📞 Contact

**Rishabh Uke**
- 📧 Email: rishabhuke14@gmail.com
- 🔗 LinkedIn: linkedin.com/in/rishabh-uke-7ab08628b/
- 🐙 GitHub: github.com/rishabhuke

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built with ❤️ for Indian Farmers</strong><br>
  94% Accuracy • 38 Disease Classes • Free Forever
</div>