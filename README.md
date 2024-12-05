
# Scalable Machine Learning API

## Overview
This project provides a **Scalable Machine Learning API** designed for **non-experts**. It simplifies the process of training machine learning models tailored to specific datasets, making machine learning accessible and efficient.

## Key Features
- **User-Friendly ML API**: Train machine learning models effortlessly without requiring deep expertise.
- **Automated Model Selection**: Fits multiple machine learning models to the provided datasets and saves the best-performing model in an **SQLite database**.
- **Seamless Deployment**: Built with **Docker** for deployment and supports **Minikube** for **Kubernetes** integration.

## Getting Started

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed.
- (Optional) [Minikube](https://minikube.sigs.k8s.io/docs/start/) for Kubernetes deployment.

---

### Run with Docker
1. Clone the repository:
   ```bash
   git clone https://github.com/ParhamMoradi98/FastAPI--ML--Microservice/tree/main
   cd ml-api
   ```

2. Build the Docker image:
   ```bash
   docker build -t ml-api .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8000:8000 ml-api
   ```

4. Access the API at:  
   **http://localhost:8000**

---

### Kubernetes Deployment (Minikube)
1. Start Minikube:
   ```bash
   minikube start
   ```

2. Deploy the application:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

3. Expose the service:
   ```bash
   kubectl port-forward service/ml-api-service 8000:8000
   ```

4. Access the API at:  
   **http://localhost:8000**

---

### SQLite Integration
The SQLite database is used to store the best-performing models. Ensure persistent storage is configured in production environments.
