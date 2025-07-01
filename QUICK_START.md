# 🚀 Quick Start Guide - Virtual Glasses Try-On

Get up and running with the Virtual Glasses Try-On system in just a few minutes!

## ⚡ Fastest Start (Recommended for beginners)

### Option 1: Using Docker (Easiest)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Generative-AI-Based-Virtual-Glasses-Try-On

# 2. Build and run with Docker
docker-compose up --build virtual-tryon

# 3. View results in demo/output/ directory
```

### Option 2: Local Setup

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd Generative-AI-Based-Virtual-Glasses-Try-On

# 2. Setup environment (installs dependencies and creates directories)
python setup.py

# 3. Run the complete pipeline
python cli.py pipeline

# 4. Check results
python cli.py demo --interactive
```

## 📋 What Happens During Setup

The system will automatically:

1. **Download Dataset** 📥
   - Downloads SCUT-FBP5500 selfie dataset from Google Drive
   - Stores ~5,000 face images in PostgreSQL database

2. **Process Images** 🖼️
   - Detects faces in selfie images
   - Removes backgrounds from glasses images
   - Calculates quality scores

3. **Initialize AI Models** 🤖
   - Downloads Meta's SAM model (~2.5GB)
   - Loads DINOv2 for feature extraction

4. **Run Demo** 🎯
   - Creates virtual try-on combinations
   - Saves results to `demo/output/`

## 🎮 Interactive Commands

Use the CLI for easy interaction:

```bash
# Show system status
python cli.py status

# Run pipeline with custom settings
python cli.py pipeline --demo-selfies 5 --demo-glasses 5

# Start training
python cli.py train --epochs 20

# Launch Jupyter for experiments
python cli.py notebook

# Clean up files
python cli.py clean --data --logs
```

## 📊 Expected Timeline

| Step | Time | Description |
|------|------|-------------|
| Setup | 2-5 min | Environment and dependencies |
| Dataset Download | 5-15 min | Download and store dataset |
| Image Preprocessing | 10-30 min | Process all images |
| Model Initialization | 5-10 min | Download and load AI models |
| Demo Generation | 2-5 min | Create try-on examples |

**Total: 25-65 minutes** (depending on internet speed and hardware)

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.10+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Internet**: Required for downloads

### Recommended Setup
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: SSD with 20GB+ free space

## 📁 Key Directories After Setup

```
📦 Project Structure
├── 📁 data/
│   ├── 📁 raw/           # Original downloaded images
│   └── 📁 processed/     # Preprocessed images
├── 📁 demo/
│   └── 📁 output/        # ✨ Your try-on results here!
├── 📁 models/
│   └── 📁 .cache/        # Downloaded AI models
└── 📁 logs/              # System logs
```

## 🎯 Quick Verification

After setup, verify everything works:

```bash
# 1. Check system status
python cli.py status

# 2. Run a quick demo
python cli.py demo

# 3. Look for output files
ls demo/output/

# 4. View results
open demo/output/comparison_grid.jpg  # macOS
# or
xdg-open demo/output/comparison_grid.jpg  # Linux
```

## 🚨 Common Issues & Solutions

### Issue: "Database connection failed"
```bash
# Solution: Check network connectivity
ping 152.53.12.68
```

### Issue: "CUDA out of memory"
```bash
# Solution: Use CPU mode or reduce batch size
python cli.py pipeline --demo-selfies 2 --demo-glasses 2
```

### Issue: "Model download failed"
```bash
# Solution: Clear cache and retry
python cli.py clean --models
python cli.py pipeline
```

### Issue: "Permission denied"
```bash
# Solution: Check file permissions
chmod +x setup.py cli.py
```

## 🎓 Next Steps

Once the system is running:

1. **Explore Results** 🔍
   - Check `demo/output/` for generated images
   - Open `comparison_grid.jpg` to see results

2. **Experiment** 🧪
   - Launch Jupyter: `python cli.py notebook`
   - Open `notebooks/experiments.ipynb`
   - Try different combinations

3. **Customize** ⚙️
   - Modify `config/settings.py` for different parameters
   - Train with your own data using `python cli.py train`

4. **Deploy** 🚀
   - Use Docker for production deployment
   - Scale with `docker-compose`

## 💡 Pro Tips

- **Use Docker** for the most reliable setup
- **Monitor logs** in `logs/application.log` for debugging
- **Start small** with limited images during development
- **Use GPU** for faster processing if available
- **Check status** regularly with `python cli.py status`

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: `tail -f logs/application.log`
2. **Run diagnostics**: `python cli.py status`
3. **Review documentation**: Open `README.md`
4. **Test components**: `python tests/test_system.py`

## 🎉 Success!

You should now have:
- ✅ Working Virtual Glasses Try-On system
- ✅ Sample try-on results in `demo/output/`
- ✅ Jupyter environment for experiments
- ✅ Complete AI pipeline ready for use

**Happy experimenting with virtual glasses try-on! 🥽✨**

---

> **Note**: This is an academic project for MSc Applied Data Science. Focus on learning the concepts and experimenting with different approaches!