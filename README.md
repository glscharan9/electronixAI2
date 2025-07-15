# Electronix AI - Sentiment Analysis Service

This is an end-to-end microservice for binary sentiment analysis.

## Setup & Run

1.  **(Optional) Fine-Tune Model**: To update the model with your own data, place it in `data/data.jsonl` and run:
    ```bash
    pip install -r backend/requirements.txt
    python finetune.py --data data/data.jsonl
    ```
2.  **Build and Start Services**:
    ```bash
    docker-compose up --build
    ```
3.  **Access**:
    -   **Frontend**: `http://localhost:3000`
    -   **API Docs**: `http://localhost:8000/docs`

## Design Decisions

-   **Backend**: FastAPI was chosen for its performance and auto-generated API docs.
-   **Frontend**: React was used for its component-based architecture.
-   **Model**: The default model is `distilbert-base-uncased-finetuned-sst-2-english`, offering a great balance of speed and accuracy.
-   **Containerization**: Docker Compose enables a consistent, single-command setup on any CPU-only machine.

## API Documentation

### POST /predict

Analyzes the sentiment of a given text string.

**Request Body:**
```json
{
  "text": "This is a wonderful experience!"
}