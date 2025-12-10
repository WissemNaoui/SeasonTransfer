# Deployment Guide for SeasonsGAN

## Overview

This guide covers deploying the trained SeasonsGAN model to production environments.

## Local Deployment (Development)

### Option 1: Streamlit (Easiest)

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

Visit: `http://localhost:8501`

**Pros:**
- Simple, no infrastructure needed
- Great for demos and sharing with colleagues
- Built-in file upload and download

**Cons:**
- Not ideal for high-traffic production
- Single-threaded

### Option 2: FastAPI (Recommended for Production)

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit: `http://localhost:8000/docs`

**Pros:**
- RESTful API, easy to integrate
- Built-in async, can handle concurrent requests
- OpenAPI documentation
- Scalable

**Cons:**
- Requires separate UI or API calls
- Slightly more complex setup

### Option 3: Docker (Production-Ready)

```bash
docker-compose up --build
```

Services:
- FastAPI: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

**Pros:**
- Reproducible across environments
- Easy to scale
- Can use GPU support

**Cons:**
- Requires Docker/Docker-Compose
- More complex initial setup

## Cloud Deployment

### AWS

#### Option A: EC2 + Docker

1. Launch EC2 instance (GPU: `g4dn.xlarge` or similar)
2. SSH into instance
3. Install Docker and clone repo
4. Run:
   ```bash
   docker-compose up --build
   ```
5. Use AWS Security Groups to expose ports 8000 and 8501
6. (Optional) Set up Application Load Balancer (ALB) for HTTPS/SSL

#### Option B: AWS App Runner (Easier)

1. Push Docker image to ECR
2. Create App Runner service
3. Point to ECR image
4. App Runner handles scaling automatically

#### Option C: SageMaker Endpoint (Most AWS-Native)

1. Create SageMaker notebook instance
2. Train or import model
3. Deploy to SageMaker Endpoint
4. Call endpoint from API Gateway

### Google Cloud Platform (GCP)

#### Option A: Cloud Run (Simplest)

```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/seasongan

gcloud run deploy seasongan \
  --image gcr.io/PROJECT-ID/seasongan \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

**Pros:** Auto-scaling, pay-per-use, no container management

#### Option B: Compute Engine + Docker

Similar to AWS EC2 approach. Launch VM, install Docker, run containers.

#### Option C: Vertex AI

Use Vertex AI custom training and prediction endpoints.

### Azure

#### Option A: Azure Container Instances (ACI)

```bash
az acr build --registry myregistry --image seasongan:latest .

az container create \
  --resource-group mygroup \
  --name seasongan \
  --image myregistry.azurecr.io/seasongan:latest \
  --ports 8000 8501
```

#### Option B: Azure App Service (with Docker)

Push image to Azure Container Registry, create App Service, configure to pull image.

## Scaling Considerations

### Single Instance
- Suitable for development/testing
- Max ~10-20 concurrent users (GPU bound)

### Load Balancing
- Use NGINX or cloud provider's load balancer
- Route requests to multiple API instances
- Each instance processes one image at a time

### Batch Processing
- For bulk image transformations
- Create a job queue (Redis, RabbitMQ)
- Scale workers as needed

### Edge Deployment (Optional)
- Use ONNX export for lighter inference
- Deploy to edge devices for offline inference
- Reduces latency and bandwidth

## Monitoring & Logging

### Local
```bash
mlflow ui --backend-store-uri ./mlruns
```

### Cloud
- AWS CloudWatch
- GCP Cloud Logging
- Azure Monitor

## Cost Estimation

### AWS
- **EC2 g4dn.xlarge** (GPU): ~$0.52/hr
- **EC2 t3.medium** (CPU): ~$0.04/hr
- **Data transfer**: ~$0.02/GB

### GCP
- **Cloud Run**: $0.00003 per request (first 2M free per month)
- **Compute Engine**: Variable

### Azure
- **Container Instances**: ~$0.013/hour
- **App Service**: ~$12/month (basic tier)

## Security Best Practices

1. **Authentication:** Add API key or OAuth2
2. **Rate Limiting:** Prevent abuse
3. **HTTPS:** Always use SSL/TLS
4. **Input Validation:** Check image size/format
5. **Model Updates:** Use versioning

## Example: Deploying to Cloud Run (GCP)

```bash
# 1. Build Docker image
docker build -t seasongan:latest .

# 2. Tag for GCP
docker tag seasongan:latest gcr.io/MY-PROJECT/seasongan:latest

# 3. Push to GCP Container Registry
docker push gcr.io/MY-PROJECT/seasongan:latest

# 4. Deploy
gcloud run deploy seasongan \
  --image gcr.io/MY-PROJECT/seasongan:latest \
  --region us-central1 \
  --memory 4Gi \
  --timeout 300 \
  --set-env-vars DEVICE=cpu  # Change to gpu if available
```

The service will be available at:
```
https://seasongan-[random-id].run.app
```

## Next Steps

1. **Test the deployment** with sample images
2. **Set up monitoring** for errors and performance
3. **Configure auto-scaling** based on traffic
4. **Add authentication** for production
5. **Document the API** for end users
