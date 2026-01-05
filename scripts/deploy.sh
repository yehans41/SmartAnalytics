#!/bin/bash

# Smart Analytics Platform Deployment Script
#
# This script automates the deployment of the Smart Analytics platform
# Usage: ./scripts/deploy.sh [environment]
#   environment: dev, staging, prod (default: dev)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-dev}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Smart Analytics Deployment${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_status "Docker found"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_status "Docker Compose found"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    print_status "Python 3 found"

    echo ""
}

# Setup environment
setup_environment() {
    echo "Setting up environment..."

    cd "$PROJECT_ROOT"

    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found, creating from template..."
        cat > .env << EOF
# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=smartanalytics
MYSQL_PASSWORD=smartpass123
MYSQL_DATABASE=smartanalytics_db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501
EOF
        print_status ".env file created"
    else
        print_status ".env file exists"
    fi

    echo ""
}

# Build Docker images
build_images() {
    echo "Building Docker images..."

    cd "$PROJECT_ROOT"

    docker-compose build
    print_status "Docker images built successfully"

    echo ""
}

# Start services
start_services() {
    echo "Starting services..."

    cd "$PROJECT_ROOT"

    # Stop existing containers
    docker-compose down 2>/dev/null || true

    # Start services
    docker-compose up -d

    print_status "Services started"

    # Wait for MySQL to be ready
    echo "Waiting for MySQL to be ready..."
    sleep 10

    # Check service health
    check_services

    echo ""
}

# Check service health
check_services() {
    echo "Checking service health..."

    # Check MySQL
    if docker-compose exec -T mysql mysqladmin ping -h localhost -u root -prootpassword &> /dev/null; then
        print_status "MySQL is healthy"
    else
        print_warning "MySQL is not ready yet"
    fi

    # Check API
    sleep 5
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "API is healthy"
    else
        print_warning "API is not ready yet (this may take a few more seconds)"
    fi

    # Check MLflow
    if curl -f http://localhost:5000/health &> /dev/null; then
        print_status "MLflow is healthy"
    else
        print_warning "MLflow is not ready yet (this may take a few more seconds)"
    fi

    echo ""
}

# Run database migrations
run_migrations() {
    echo "Running database migrations..."

    cd "$PROJECT_ROOT"

    # Check if init script exists
    if [ -f "scripts/init_db.sql" ]; then
        docker-compose exec -T mysql mysql -u root -prootpassword smartanalytics_db < scripts/init_db.sql
        print_status "Database migrations completed"
    else
        print_warning "No migration script found"
    fi

    echo ""
}

# Display deployment info
show_deployment_info() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "Access the platform at:"
    echo ""
    echo -e "  ${GREEN}API:${NC}           http://localhost:8000"
    echo -e "  ${GREEN}API Docs:${NC}      http://localhost:8000/docs"
    echo -e "  ${GREEN}Dashboard:${NC}     http://localhost:8501"
    echo -e "  ${GREEN}MLflow UI:${NC}     http://localhost:5000"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f [service]"
    echo ""
    echo "To stop services:"
    echo "  docker-compose down"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    setup_environment
    build_images
    start_services

    # Only run migrations in dev/staging
    if [ "$ENVIRONMENT" != "prod" ]; then
        run_migrations
    fi

    show_deployment_info
}

# Run main function
main
