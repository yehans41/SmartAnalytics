#!/bin/bash

# Smart Analytics Platform - Quickstart Script
#
# This script sets up and runs the complete platform end-to-end
# Usage: ./scripts/quickstart.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Smart Analytics - Quickstart${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

cd "$PROJECT_ROOT"

# Step 1: Start Docker services
echo -e "${GREEN}Step 1/6: Starting Docker services...${NC}"
make docker-up
sleep 15
echo ""

# Step 2: Ingest data
echo -e "${GREEN}Step 2/6: Ingesting sample data...${NC}"
echo "(This will download and load NYC taxi data - may take a few minutes)"
make ingest
echo ""

# Step 3: Process data
echo -e "${GREEN}Step 3/6: Processing and cleaning data...${NC}"
make process
echo ""

# Step 4: Engineer features
echo -e "${GREEN}Step 4/6: Engineering features...${NC}"
make features
echo ""

# Step 5: Train models
echo -e "${GREEN}Step 5/6: Training ML models...${NC}"
echo "(This will train regression and classification models - may take 5-10 minutes)"
make train-regression
echo ""

# Step 6: Show completion
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}✅ Quickstart Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "The Smart Analytics platform is now ready!"
echo ""
echo -e "${GREEN}Access the services:${NC}"
echo ""
echo "  📊 Dashboard:   http://localhost:8501"
echo "  🔬 API Docs:    http://localhost:8000/docs"
echo "  📈 MLflow:      http://localhost:5000"
echo ""
echo -e "${GREEN}Try these commands:${NC}"
echo ""
echo "  # View logs"
echo "  docker-compose logs -f api"
echo ""
echo "  # Make a prediction"
echo "  curl http://localhost:8000/health"
echo ""
echo "  # Stop services"
echo "  make docker-down"
echo ""
echo -e "${YELLOW}Note: The dashboard may take a few seconds to start.${NC}"
echo ""
