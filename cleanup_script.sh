#!/bin/bash

echo "🧹 Cleaning up Virtual Glasses Try-On project..."

# Remove unnecessary diagnostic scripts
rm -f diagnose_database_data.py
rm -f fix_database_schema.py
rm -f quick_fix_demo_data.py

# Remove log files
rm -f app.log
rm -f pipeline.log
rm -rf logs/setup_verification_report.json

# Remove generated notebooks (they can be recreated)
rm -f notebooks/experiments.ipynb

# Clean up any Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Create necessary directories
mkdir -p data/{raw,processed}
mkdir -p demo/{selfies,output}
mkdir -p models/{sam_checkpoints,dino_checkpoints}
mkdir -p logs
mkdir -p notebooks/outputs

echo "✅ Cleanup completed!"
echo "📁 Project structure cleaned and organized"