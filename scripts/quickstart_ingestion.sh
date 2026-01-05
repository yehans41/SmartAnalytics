#!/bin/bash

# Quick Start Script for Phase 1: Data Ingestion

set -e

echo "=========================================="
echo "NYC Taxi Data Ingestion - Quick Start"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    source venv/bin/activate
fi

# Check if MySQL is running
echo -e "\n${YELLOW}Checking MySQL connection...${NC}"
if mysql -h ${MYSQL_HOST:-localhost} -u ${MYSQL_USER:-smartanalytics} -p${MYSQL_PASSWORD:-changeme123} -e "SELECT 1" &> /dev/null; then
    echo -e "${GREEN}✓ MySQL is running${NC}"
else
    echo -e "${RED}✗ MySQL is not running or credentials are wrong${NC}"
    echo "Starting MySQL with Docker..."
    docker-compose up -d mysql
    echo "Waiting for MySQL to be ready..."
    sleep 10
fi

# Initialize database if needed
echo -e "\n${YELLOW}Checking database schema...${NC}"
if mysql -h ${MYSQL_HOST:-localhost} -u ${MYSQL_USER:-smartanalytics} -p${MYSQL_PASSWORD:-changeme123} ${MYSQL_DATABASE:-smartanalytics_db} -e "SHOW TABLES LIKE 'raw_taxi_trips'" | grep -q "raw_taxi_trips"; then
    echo -e "${GREEN}✓ Database tables exist${NC}"
else
    echo -e "${YELLOW}Initializing database schema...${NC}"
    mysql -h ${MYSQL_HOST:-localhost} -u ${MYSQL_USER:-smartanalytics} -p${MYSQL_PASSWORD:-changeme123} ${MYSQL_DATABASE:-smartanalytics_db} < scripts/init_db.sql
    echo -e "${GREEN}✓ Database initialized${NC}"
fi

# Option menu
echo -e "\n${YELLOW}Select an option:${NC}"
echo "1) Download and ingest sample data (1 month, 10K rows) - RECOMMENDED FOR TESTING"
echo "2) Download and ingest full month (1 month, ~3M rows)"
echo "3) Download and ingest multiple months (Jan-Mar 2023)"
echo "4) Just test download (no database insertion)"
echo "5) Run unit tests"

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running sample ingestion (10K rows)...${NC}"
        python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1 --sample 10000 --mode replace
        ;;
    2)
        echo -e "\n${GREEN}Running full month ingestion...${NC}"
        python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1 --mode replace
        ;;
    3)
        echo -e "\n${GREEN}Running multi-month ingestion (Jan-Mar 2023)...${NC}"
        python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 3 --mode replace
        ;;
    4)
        echo -e "\n${GREEN}Testing download only...${NC}"
        python -m src.ingestion.download
        ;;
    5)
        echo -e "\n${GREEN}Running unit tests...${NC}"
        pytest tests/unit/test_ingestion.py -v
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

# Verify ingestion
if [[ $choice -eq 1 ]] || [[ $choice -eq 2 ]] || [[ $choice -eq 3 ]]; then
    echo -e "\n${YELLOW}Verifying ingestion...${NC}"
    ROW_COUNT=$(mysql -h ${MYSQL_HOST:-localhost} -u ${MYSQL_USER:-smartanalytics} -p${MYSQL_PASSWORD:-changeme123} ${MYSQL_DATABASE:-smartanalytics_db} -N -e "SELECT COUNT(*) FROM raw_taxi_trips")
    echo -e "${GREEN}✓ Total rows in database: ${ROW_COUNT}${NC}"

    echo -e "\n${YELLOW}Sample data:${NC}"
    mysql -h ${MYSQL_HOST:-localhost} -u ${MYSQL_USER:-smartanalytics} -p${MYSQL_PASSWORD:-changeme123} ${MYSQL_DATABASE:-smartanalytics_db} -e "SELECT * FROM raw_taxi_trips LIMIT 3"
fi

echo -e "\n${GREEN}=========================================="
echo "Phase 1 Complete!"
echo "==========================================${NC}"
echo -e "\nNext steps:"
echo "  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "  2. Start Phase 2: make process"
echo "  3. View data in MySQL: make db-shell"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo "  make ingest              # Run ingestion with defaults"
echo "  make mlflow-ui           # View experiments (after training)"
echo "  make test                # Run all tests"
