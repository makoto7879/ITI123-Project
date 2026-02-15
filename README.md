---
title: Email Spam Detector
emoji: 📧
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

ImprovedSPAMGuru - Email Spam Detection System
A deep learning-based spam detection system using Bidirectional GRU with Attention mechanism, achieving 97.82% test accuracy.
🎯 Overview
This project implements an advanced email spam classifier that can distinguish between legitimate emails (HAM) and spam with high accuracy. The model uses natural language processing and deep learning to analyze email content and detect spam patterns including phishing attempts, prize scams, and money-based fraud.
📊 Performance

Test Accuracy: 97.82%
Validation Accuracy: 98.62%
Test Loss: 0.0627
Dataset: Spam Assassin (5,796 emails)

🏗️ Model Architecture

Type: Bidirectional GRU + Attention Mechanism
Embedding Dimension: 256
Hidden Size: 256
Layers: 2-layer BiGRU
Dropout: 0.3
Vocabulary Size: ~15,000 words
Total Parameters: ~5.9M

✨ Features
Advanced Detection Capabilities:

✅ Phishing pattern recognition (urgent + verify + URL combinations)
✅ Prize scam detection (large money amounts + winning language)
✅ URL domain analysis
✅ Money amount extraction
✅ Special token vocabulary for spam patterns

Special Tokens:

phishing - Phishing email indicators
largemoney - Large dollar amounts ($10,000+)
money - Smaller amounts ($100+)
prizescam - Prize/lottery scam patterns

🚀 Quick Start
Training the Model

Open the notebook in Google Colab:

Upload TEST_TEST_14_Feb_2026_WanDB.ipynb
Upload spam_assassin.csv dataset


Install dependencies:

python   # Already included in notebook
   - PyTorch
   - NLTK
   - BeautifulSoup4
   - Pandas, NumPy
   - scikit-learn
   - Weights & Biases (optional)

Run all cells sequentially:

The notebook handles everything automatically
Training takes ~10-15 minutes on GPU (T4)


Download trained models:

spam_model.pth (78MB) - Model weights
spam_config.pth (2MB) - Vocabulary and configuration



Deploying the Web App

Required files:

   app.py
   spam_model.pth
   spam_config.pth
   requirements.txt

Local deployment:

bash   pip install -r requirements.txt
   python app.py

Hugging Face Spaces deployment:

Create a new Space on Hugging Face
Upload all files
Platform automatically builds and deploys


Access the demo:

Live demo: https://huggingface.co/spaces/makoto7879/Email-Spam-Detector



📦 Project Structure
├── 14_Feb_2026_WanDB.ipynb             # Main training notebook
├── app.py                              # Gradio web interface
├── requirements.txt                    # Python dependencies
├── spam_model.pth                      # Trained model weights
├── spam_config.pth                     # Model configuration
└── README.md                           # This file
🛠️ Technical Details
Preprocessing Pipeline:

HTML tag removal (BeautifulSoup)
URL domain extraction
Email/phone number replacement
Money amount detection
Phishing pattern detection
Prize scam detection
Text normalization
NLTK tokenization
Vocabulary encoding

Training Configuration:

Optimizer: Adam
Learning Rate: 0.001
Epochs: 10
Loss Function: BCELoss (Binary Cross-Entropy)
Device: CUDA (GPU) or CPU fallback

Data Split:

Training: 70%
Validation: 15%
Test: 15%

📈 Experiment Tracking
The project uses Weights & Biases for experiment tracking:

Real-time training visualization
Hyperparameter logging
Model comparison
Metric tracking (train loss, val loss, val accuracy)

🎨 Web Interface Features

Simple text input for email content
Pre-loaded spam/ham examples
Real-time classification (<2 seconds)
Confidence scores
Detailed analysis showing:

Detected signals (phishing, money, URLs)
Token preview
Classification reasoning



📊 Example Classifications
SPAM Examples:

✅ "You've won $2,500,000! Claim now: http://luckyclaim.net" → 🚨 SPAM
✅ "URGENT: Account suspended! Verify: http://secure-bank.xyz" → 🚨 SPAM
✅ "FREE iPhone 15! Click here now!" → 🚨 SPAM

HAM Examples:

✅ "Hi team, meeting tomorrow at 2 PM" → ✅ HAM
✅ "Thanks for the update on the contract" → ✅ HAM

🔧 Requirements
txttorch>=2.0.0
gradio>=4.0.0
nltk>=3.8
beautifulsoup4>=4.11.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
wandb>=0.15.0 (optional)
🤝 Credits

Dataset: Spam Assassin Public Corpus
Framework: PyTorch
Deployment: Gradio + Hugging Face Spaces
AI Assistance: Claude AI, Grok AI

📝 License
This project is for educational purposes as part of the ITI123 Generative AI & Deep Learning course.
🎓 Academic Context
Course: ITI123 Generative AI & Deep Learning
Project Type: Model Development Focus
Institution: Nanyang Polytechnic
Semester: 2025S2

Note: This is a proof-of-concept spam detection system. For production use, consider:

Regular model retraining with new spam patterns
Integration with email clients via API
User feedback mechanism for continuous improvement
Multi-language support
Batch processing capabilities

**Built with ❤️ using PyTorch and Gradio**

*For issues or questions, please check the troubleshooting section or review your training notebook's configuration.*
