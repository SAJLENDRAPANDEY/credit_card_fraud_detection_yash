# 🚀 Quick Start Guide

Get the Fraud Detection Suite running in minutes!

---

## ⚡ 5-Minute Quick Start

### 1. **Download & Navigate**
```bash
# Clone or download the project
cd fraud-detection
```

### 2. **Install Python Dependencies**
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. **Run the App**
```bash
streamlit run app.py
```

That's it! 🎉

Your app will open at: **http://localhost:8501**

---

## 🛠️ System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Disk Space**: 500MB
- **OS**: Windows, macOS, or Linux

### Check Python Version
```bash
python --version
```

---

## 📥 Installation Methods

### Method 1: pip (Easiest)
```bash
pip install -r requirements.txt
```

### Method 2: Conda
```bash
conda create -n fraud-detection python=3.9
conda activate fraud-detection
pip install -r requirements.txt
```

### Method 3: Docker (No Python needed)
```bash
# Build image
docker build -t fraud-detection .

# Run container
docker run -p 8501:8501 fraud-detection
```

### Method 4: Docker Compose
```bash
docker-compose up
```

---

## 🎯 First Run Checklist

- [ ] Python installed (3.8+)
- [ ] Requirements installed
- [ ] Model files present (`xgb_model.pkl`, `xgb_scaler.pkl`)
- [ ] App running without errors
- [ ] Browser opened to localhost:8501

---

## 📊 Test the App

### Using Sample Data
1. Open the app
2. Go to **Detection** tab
3. Upload sample CSV
4. Click **Analyze Transactions**
5. View results and download

### Create Sample Data
```python
import pandas as pd
import numpy as np

# Create sample data with required 30 features
data = {
    'Time': range(100),
    'V1': np.random.randn(100),
    'V2': np.random.randn(100),
    # ... add V3-V28 ...
    'Amount': np.random.uniform(0, 1000, 100)
}

df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)
```

---

## 🔧 Common Commands

### Start App
```bash
streamlit run app.py
```

### Clear Cache
```bash
streamlit cache clear
```

### Run with Custom Port
```bash
streamlit run app.py --server.port 8502
```

### Enable Logger
```bash
streamlit run app.py --logger.level=debug
```

### Run in Production Mode
```bash
streamlit run app.py --server.headless true --server.port 8501
```

---

## 🆘 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Issue: "xgb_model.pkl not found"

**Solution:**
1. Ensure model files are in the app directory
2. Check file names are exactly correct (case-sensitive)
3. Train a new model: `python train_model.py`

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process using port
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

### Issue: App runs but won't load data

**Solution:**
- Check CSV format (30 columns required)
- Verify column names match
- Ensure no missing values or corrupted data

### Issue: Slow predictions on large files

**Solution:**
- Process in smaller batches
- Enable caching (already done)
- Use stronger hardware
- Reduce dataset size

---

## 📚 Next Steps

1. **Read the Documentation**
   - Check `README.md` for full details
   - Review `DEPLOYMENT_GUIDE.md` for production

2. **Train Custom Model**
   ```bash
   python train_model.py
   ```

3. **Deploy to Cloud**
   - Streamlit Cloud (easiest)
   - Docker (Docker Hub)
   - Hugging Face Spaces

4. **Customize the App**
   - Edit colors in `app.py`
   - Change model parameters
   - Add new features

---

## 💡 Tips for Success

✅ **Do's**
- Keep model files in app directory
- Use clean, validated CSV data
- Monitor prediction logs
- Update requirements.txt regularly
- Test before deployment

❌ **Don'ts**
- Don't modify model pickle files manually
- Don't use corrupted CSV files
- Don't skip requirements installation
- Don't commit large data files to git
- Don't expose API keys in code

---

## 🚀 Deployment Quick Links

| Platform | Setup Time | Cost | Easiest? |
|----------|-----------|------|----------|
| **Local** | 5 min | Free | ✅ Yes |
| **Streamlit Cloud** | 10 min | Free | ✅ Yes |
| **Docker** | 15 min | Free | ⚠️ Medium |
| **HF Spaces** | 10 min | Free | ✅ Yes |
| **AWS** | 30 min | ~$10/mo | ❌ Complex |

**Recommended**: Start with **Streamlit Cloud** for easiest deployment!

---

## 📧 Need Help?

- 📖 Read `README.md`
- 📋 Check `DEPLOYMENT_GUIDE.md`
- 💬 Contact: sajlendrapandey2022@gmail.com
- 🐙 GitHub: https://github.com/SAJLENDRAPANDEY

---

## 🎓 Learning Path

1. **Run locally** (this guide)
2. **Understand the code** (`app.py`)
3. **Train a model** (`train_model.py`)
4. **Deploy to cloud** (see deployment guide)
5. **Monitor in production**
6. **Optimize performance**

---

## ✨ Key Features at a Glance

- 🔍 **Real-time Fraud Detection**
- 📊 **Interactive Dashboard**
- 📈 **Advanced Analytics**
- 📥 **Batch Processing**
- 💾 **Export Results**
- 🎨 **Beautiful UI**
- 🚀 **Production Ready**

---

**Enjoy exploring fraud detection! 💳🔒**

Last Updated: 2024 | Made with ❤️ by Yash Sajlendra Pandey
