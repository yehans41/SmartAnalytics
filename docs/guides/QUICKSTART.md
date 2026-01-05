# Quick Start Guide

## Step 1: Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Set Up Database

```bash
# Start MySQL with Docker
docker-compose up -d mysql

# Wait for MySQL to start
sleep 10
```

## Step 3: Run Ingestion

```bash
# Ingest sample data (10K rows - fast)
make ingest

# Or use the interactive script
./scripts/quickstart_ingestion.sh
```

## Troubleshooting

### "python: No such file or directory"
✅ Fixed! Makefile now uses python3

### "No module named 'src'"
Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### "MySQL connection failed"
Start MySQL:
```bash
docker-compose up -d mysql
sleep 10
```
