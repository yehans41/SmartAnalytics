#!/bin/bash

# Smart Analytics Setup Script

set -e

echo "==================================="
echo "Smart Analytics Setup"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Setup pre-commit hooks
echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
pre-commit install

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}.env file created. Please update with your configuration.${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data/raw data/processed data/features
mkdir -p logs
mkdir -p mlruns
mkdir -p models/registry models/artifacts
mkdir -p notebooks
echo -e "${GREEN}Directories created${NC}"

# Check MySQL connection
echo -e "\n${YELLOW}Checking MySQL connection...${NC}"
if command -v mysql &> /dev/null; then
    echo -e "${GREEN}MySQL client found${NC}"
else
    echo -e "${RED}MySQL client not found. Please install MySQL.${NC}"
fi

# Check Docker
echo -e "\n${YELLOW}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker found${NC}"
    echo "You can start services with: docker-compose up -d"
else
    echo -e "${YELLOW}Docker not found. Docker is optional but recommended.${NC}"
fi

# Setup complete
echo -e "\n${GREEN}==================================="
echo "Setup Complete!"
echo "===================================${NC}"
echo -e "\nNext steps:"
echo "1. Update .env file with your configuration"
echo "2. Start MySQL (docker-compose up -d mysql OR use local MySQL)"
echo "3. Run database initialization: mysql < scripts/init_db.sql"
echo "4. Run the pipeline: make pipeline"
echo -e "\nUseful commands:"
echo "  make help       - Show all available commands"
echo "  make pipeline   - Run full ML pipeline"
echo "  make test       - Run tests"
echo "  make serve      - Start API server"
echo "  make mlflow-ui  - Start MLflow UI"
