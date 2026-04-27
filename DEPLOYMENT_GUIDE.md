# 🚀 Deployment Guide

Complete guide to deploy the Fraud Detection Suite to various platforms.

---

## Option 1: Streamlit Cloud (Recommended - Easiest)

### Prerequisites
- GitHub account
- Streamlit account (free)
- Project pushed to GitHub

### Step-by-Step Deployment

#### 1. Prepare GitHub Repository
```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial fraud detection app"

# Push to GitHub
git push -u origin main
```

#### 2. Create Streamlit Cloud Account
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Sign in with GitHub
4. Grant permissions

#### 3. Deploy Application
1. Select your repository
2. Select branch: `main`
3. Select file path: `app.py`
4. Click "Deploy"

**Streamlit will:**
- Install dependencies from `requirements.txt`
- Run your app
- Provide a shareable URL

#### 4. Configuration (Optional)
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#0066FF"
backgroundColor = "#0F1419"
secondaryBackgroundColor = "#1a1f2e"
textColor = "#E0E6ED"
font = "sans serif"

[client]
showErrorDetails = false

[server]
maxUploadSize = 200
```

#### 5. Access Your App
Your app will be available at:
```
https://[your-username]-fraud-detection.streamlit.app
```

### Troubleshooting
- **Missing dependencies**: Update `requirements.txt`
- **Memory issues**: Reduce data size or use `@st.cache_resource`
- **Slow loading**: Optimize feature calculations

---

## Option 2: Docker Containerization

### Prerequisites
- Docker installed
- Docker Hub account (optional, for publishing)

### Step 1: Create Dockerfile

Already included as `Dockerfile` in the project.

### Step 2: Build Docker Image

```bash
# Build image
docker build -t fraud-detection:latest .

# Tag for Docker Hub (optional)
docker tag fraud-detection:latest your-username/fraud-detection:latest
```

### Step 3: Run Container Locally

```bash
# Run on port 8501
docker run -p 8501:8501 fraud-detection:latest

# Run with volume mount (for data files)
docker run -p 8501:8501 -v $(pwd):/app fraud-detection:latest
```

Access at: `http://localhost:8501`

### Step 4: Push to Docker Hub (Optional)

```bash
# Login
docker login

# Push
docker push your-username/fraud-detection:latest

# Pull (from anywhere)
docker pull your-username/fraud-detection:latest
```

### Docker Compose (For Multiple Services)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  fraud-detection:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_LOGGER_LEVEL=info
```

Run with:
```bash
docker-compose up
```

---

## Option 3: Hugging Face Spaces

### Prerequisites
- Hugging Face account
- GitHub repository or direct upload

### Step 1: Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Select:
   - **Owner**: Your profile
   - **Space name**: `fraud-detection`
   - **Space type**: `Streamlit`
   - **Visibility**: Public/Private

### Step 2: Upload Files

#### Option A: Git Push
```bash
# Clone the space
git clone https://huggingface.co/spaces/your-username/fraud-detection
cd fraud-detection

# Copy your files
cp ../fraud-detection/* .

# Push to HF
git add .
git commit -m "Add fraud detection app"
git push
```

#### Option B: Web Upload
1. Go to your Space settings
2. Click "Upload files"
3. Select `app.py`, `requirements.txt`, and model files

### Step 3: Configure Space

Create `requirements.txt` in the space (if not present):
```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
imbalanced-learn==0.11.0
plotly==5.18.0
```

### Step 4: Access Your App

Your app will be available at:
```
https://huggingface.co/spaces/your-username/fraud-detection
```

### File Size Limits
- **Free tier**: 10GB
- **Paid tier**: Larger limits

### Tips for HF Spaces
- Use `.gitignore` to exclude large files
- Use Git LFS for large datasets
- Keep model files under 500MB

---

## Option 4: AWS Deployment

### Using EC2 + SystemD

#### Step 1: Launch EC2 Instance
- Choose: Ubuntu 20.04 LTS
- Instance type: t2.micro (free tier)
- Security group: Allow HTTP (80) & HTTPS (443)

#### Step 2: Connect via SSH
```bash
ssh -i your-key.pem ec2-user@your-instance.com
```

#### Step 3: Install Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv -y

# Clone your repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 4: Configure SystemD Service
Create `/etc/systemd/system/fraud-detection.service`:
```ini
[Unit]
Description=Fraud Detection Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/fraud-detection
Environment="PATH=/home/ubuntu/fraud-detection/venv/bin"
ExecStart=/home/ubuntu/fraud-detection/venv/bin/streamlit run app.py --server.port=8501

[Install]
WantedBy=multi-user.target
```

#### Step 5: Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable fraud-detection
sudo systemctl start fraud-detection

# Check status
sudo systemctl status fraud-detection
```

#### Step 6: Setup Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Option 5: Heroku Deployment (Deprecated - Use Alternatives)

Since Heroku free tier ended, consider:
- Render.com
- Railway.app
- Fly.io

### Using Render.com

#### Step 1: Connect GitHub
1. Create account at [render.com](https://render.com)
2. Click "New +"
3. Select "Web Service"
4. Connect GitHub repository

#### Step 2: Configure
- **Name**: `fraud-detection`
- **Environment**: Python 3
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `streamlit run app.py --server.port 10000`

#### Step 3: Deploy
- Set environment variables (if needed)
- Click "Create Web Service"
- Your app will be live at provided URL

---

## Production Checklist

- [ ] Test app locally
- [ ] Update `requirements.txt`
- [ ] Create `.gitignore`
- [ ] Add environment variables for secrets
- [ ] Test with sample data
- [ ] Setup monitoring/logging
- [ ] Configure auto-restart
- [ ] Setup backup strategy
- [ ] Test error handling
- [ ] Monitor resource usage
- [ ] Setup alerts

---

## Performance Optimization

### For Large Files

```python
# Use caching
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

# Use session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()
```

### For Slow Predictions

```python
# Batch processing
if len(df) > 10000:
    st.warning("Large dataset. Processing in batches...")
    batch_size = 10000
    predictions = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        pred = model.predict_proba(batch)
        predictions.extend(pred)
```

### Memory Management

```bash
# Monitor memory
free -h

# Limit Streamlit memory
export STREAMLIT_CLIENT_MAX_MESSAGE_SIZE=200
```

---

## Monitoring & Logging

### Setup Logging
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("App started")
```

### Monitor Deployments

#### Streamlit Cloud
- Check "Logs" tab
- Monitor resource usage
- View deployment history

#### Docker
```bash
# View logs
docker logs container-id

# Monitor resources
docker stats
```

#### Linux Service
```bash
# View logs
journalctl -u fraud-detection -f

# Monitor
systemctl status fraud-detection
```

---

## Troubleshooting Deployment

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Update `requirements.txt` |
| `OutOfMemory` | Reduce batch size, use caching |
| `Port already in use` | Change port: `streamlit run app.py --server.port 8502` |
| `Model file not found` | Check file paths, use absolute paths |
| `Slow loading` | Enable caching with `@st.cache_resource` |

### Debug Mode
```bash
# Run with debug logging
streamlit run app.py --logger.level=debug
```

---

## Cost Comparison

| Platform | Free Tier | Cost | Best For |
|----------|-----------|------|----------|
| **Streamlit Cloud** | Yes | Free-$50/mo | Getting started |
| **HF Spaces** | Yes | Free-$9/mo | ML projects |
| **Docker Hub** | 1 repo | Free-$7/mo | Production |
| **AWS EC2** | Limited | ~$10/mo | Heavy workloads |
| **Render** | Limited | $7-15/mo | Simple apps |
| **Railway** | $5 credit | Pay as you go | Flexible |

---

## SSL/HTTPS Setup

### Let's Encrypt (Free)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Generate certificate
sudo certbot certonly --nginx -d your-domain.com

# Auto-renew
sudo systemctl enable certbot.timer
```

---

## Backup Strategy

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
mkdir -p $BACKUP_DIR

# Backup model files
tar -czf $BACKUP_DIR/models_$(date +%Y%m%d).tar.gz *.pkl

# Backup logs
tar -czf $BACKUP_DIR/logs_$(date +%Y%m%d).tar.gz logs/

# Keep last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Schedule with cron:
```bash
0 2 * * * /path/to/backup.sh
```

---

## Support & Updates

- Check deployment logs regularly
- Monitor model performance
- Keep dependencies updated
- Plan for scaling
- Document configuration

---

**For detailed platform documentation:**
- [Streamlit Cloud](https://docs.streamlit.io/streamlit-cloud)
- [Docker](https://docs.docker.com/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [AWS](https://aws.amazon.com/documentation/)

---

**Contact**: sajlendrapandey2022@gmail.com | [GitHub](https://github.com/SAJLENDRAPANDEY)
