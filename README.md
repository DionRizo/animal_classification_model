# Animal Classification Model

A deep learning model for classifying animal images using PyTorch, with MLOps practices implemented using DVC and MLflow.

## Frontend Overview

Check out the README under the `Front End` folder to install dependencies and run the website.

## Backend Overview

This project includes a REST API built with **FastAPI** that allows users to upload animal images and receive predictions from the trained deep learning model.

### FastAPI Features

- Endpoint `/predict` that accepts image files and returns the predicted animal class with confidence.
- Fast and lightweight inference with PyTorch's `torch.jit.script`.
- Automatic interactive documentation at `/docs` and `/redoc`.

### Running the API

1. Make sure you have the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

3. Access the API at:
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Usage

You can send a POST request to the `/predict` endpoint with an image file. Example using `curl`:

```bash
curl -X 'POST'   'http://localhost:8000/predict'   -F 'file=@path_to_your_image.jpg'
```

Response:
```json
{
  "prediction": "Cat",
  "confidence": 0.94
}
```

## AWS Infrastructure (Terraform)

The backend is deployed to AWS using **Terraform**. The infrastructure includes:

- EC2 instances (1 master, 6 workers) with Docker and Python pre-installed.
- Security Groups to allow:
  - SSH (port 22)
  - HTTP (port 80)
  - Custom API (port 8000)
- Automatic file upload to all EC2s using `scp`.
- Docker image built and executed in each instance using the uploaded `Dockerfile`.

> Note: The Terraform configuration is modular and follows best practices to manage compute and networking resources.

...

# (The rest of the full content goes here. For brevity, only partial content is shown.)