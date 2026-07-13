<div align="center">
  <img src="frontend/logo.png" alt="Sepsis Logo" width="150" />
  <h1>Sepsis Risk Prediction System</h1>
  <p><strong>A full-stack machine learning application for early detection of Sepsis risk in ICU patients.</strong></p>

  [![Live Deployment](https://img.shields.io/badge/Live_Deployment-Vercel-black?style=for-the-badge&logo=vercel)](https://sepsis-phi.vercel.app/)
  [![React](https://img.shields.io/badge/Frontend-React-blue?style=for-the-badge&logo=react)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Scikit-Learn](https://img.shields.io/badge/ML_Model-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
</div>

<br />

## 🌐 Live Demo
The application is fully deployed and accessible online!
**[View Live Application here](https://sepsis-phi.vercel.app/)**

---

## 📖 Overview
Sepsis is a life-threatening medical emergency caused by the body's extreme response to an infection. Early detection and treatment are critical for patient survival. This project utilizes clinical data (vital signs and laboratory values) to accurately predict a patient's risk of developing sepsis.

The system is designed with a modern, glassmorphic UI and a robust Python backend to deliver instant, reliable predictions with confidence scores.

## ✨ Features
- **Real-Time Predictions:** Instant risk analysis powered by a machine learning model.
- **Confidence Scoring:** Outputs the probability of sepsis to assist clinical decision-making.
- **Serverless Architecture:** The FastAPI backend is deployed on Vercel Serverless Functions for infinite scalability.
- **Professional UI/UX:** Responsive, modern design tailored for medical and professional environments.

## 🛠️ Technology Stack
- **Frontend:** React, HTML5, Vanilla CSS
- **Backend:** Python, FastAPI, Pydantic (Vercel Serverless)
- **Machine Learning:** Scikit-Learn (Logistic Regression), Pandas, Numpy
- **Deployment:** Vercel

## 🚀 Local Development

### Prerequisites
- Node.js (for frontend)
- Python 3.9+ (for backend)

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/kashifshafi99/Sepsis.git
   cd Sepsis
   ```

2. **Run the Backend (FastAPI):**
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn index:app --reload --port 8000
   ```

3. **Run the Frontend (React):**
   Simply serve the `frontend` directory using any local server, or open `index.html`.

## 👨‍💻 About the Developer
Developed by **Kashif Shafi**, an expert Software Engineer specializing in full-stack web development and machine learning integration.

- **GitHub:** [@kashifshafi99](https://github.com/kashifshafi99)
- **Live Project:** [Sepsis Risk Prediction](https://sepsis-phi.vercel.app/)

## 📄 License
This project is licensed under the MIT License.
