# spaCy Word Embedding API

This project implements a FastAPI application that generates word embeddings using spaCy's `en_core_web_lg` model. It was developed for Assignment 1 of Applied Generative AI.

## Features

* Generate a 300-dimensional word embedding for a query word
* Confirm whether spaCy has a vector for the word
* Check API health status
* Run locally with `uv`
* Deploy with Docker

## API Endpoints

### Root endpoint

```text
GET /
```

Returns basic API information.

### Health check

```text
GET /health
```

Returns the server and model status.

### Word embedding

```text
GET /embedding?word=apple
```

Example response:

```json
{
  "word": "apple",
  "dimension": 300,
  "has_vector": true,
  "embedding": [...]
}
```

## Run Locally

Install the dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run fastapi dev main.py
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build the Docker image:

```bash
docker build -t sps-genai .
```

Run the Docker container:

```bash
docker run --rm -d -p 8000:80 --name sps-genai-container sps-genai
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Test the word embedding endpoint directly:

```text
http://127.0.0.1:8000/embedding?word=apple
```

Stop the Docker container:

```bash
docker stop sps-genai-container
```
