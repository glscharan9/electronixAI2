# Electronix AI – Sentiment Analysis Microservice

This project is a complete, containerized microservice for binary sentiment classification (`positive` / `negative`) using Hugging Face Transformers. It includes a FastAPI backend, a React frontend, and a CLI fine-tuning script. Built as part of the Electronix AI Fullstack Intern Assignment.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed

### Run the App
```bash
docker-compose up --build
```

- Frontend: http://localhost:3000  
- API: http://localhost:8000/predict

### Sample API Usage
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"text": "Great product!"}'
```

Response:
```json
{ "label": "positive", "score": 0.98 }
```

---

## 🛠 Fine-Tuning Script

Train the model on custom data and save weights:

```bash
python finetune.py -data data.jsonl -epochs 3 -lr 3e-5
```

**Sample data.jsonl:**
```json
{"text": "Loved it!", "label": "positive"}
{"text": "Not good.", "label": "negative"}
```

- Saves fine-tuned model to `./model/`
- Automatically reloaded by API on next run

---

## 🧰 Tech Stack

- **Backend:** Python, FastAPI, Hugging Face Transformers  
- **Frontend:** React  
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- **Infra:** Docker, Docker Compose

---

## 📁 Project Structure

- `app/` – FastAPI backend  
- `frontend/` – React frontend  
- `finetune.py` – Training CLI  
- `docker-compose.yml`, `Dockerfile`, `requirements.txt`

---


---

> ✅ Built with ❤️ for the Electronix AI Fullstack Intern Assignment
