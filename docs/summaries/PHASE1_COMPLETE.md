# Phase 1: Data Ingestion - COMPLETE ✅

## What Was Built

### 1. Data Download Module ([src/ingestion/download.py](src/ingestion/download.py))

**Purpose**: Download NYC Taxi data from official TLC website

**Key Features**:
- `NYCTaxiDownloader` class for downloading parquet files
- Download single or multiple months
- Progress bars for downloads
- Automatic file verification
- No API key required (uses public NYC TLC data)

**Usage**:
```python
from src.ingestion.download import download_nyc_taxi_data

# Download single month
files = download_nyc_taxi_data(year=2023, start_month=1, end_month=1)

# Download multiple months
files = download_nyc_taxi_data(year=2023, start_month=1, end_month=3)
```

**Command Line**:
```bash
python -m src.ingestion.download  # Downloads sample (Jan 2023)
```

---

### 2. Data Ingestion Module ([src/ingestion/ingest_data.py](src/ingestion/ingest_data.py))

**Purpose**: Load data from parquet files into MySQL database

**Key Features**:
- `TaxiDataIngester` class for database operations
- Schema validation
- Data preparation and cleaning
- Batch insertion for performance
- Comprehensive logging and statistics
- Support for sampling (for testing)

**Usage**:
```python
from src.ingestion.ingest_data import run_ingestion

# Ingest sample data (10K rows)
run_ingestion(
    year=2023,
    start_month=1,
    end_month=1,
    sample_size=10000
)

# Ingest full month
run_ingestion(year=2023, start_month=1, end_month=1)
```

**Command Line**:
```bash
# Sample data (10K rows) - RECOMMENDED FOR TESTING
python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1 --sample 10000

# Full month (~3M rows)
python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 1

# Multiple months
python -m src.ingestion.ingest_data --year 2023 --start-month 1 --end-month 3

# Skip download if already have files
python -m src.ingestion.ingest_data --no-download --year 2023 --start-month 1 --end-month 1
```

---

### 3. Data Exploration Notebook ([notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb))

**Purpose**: Explore and analyze ingested data

**Includes**:
- Load data from MySQL
- Basic statistics and distributions
- Data quality checks
- Temporal patterns (hourly, daily)
- Feature correlations
- Visualization of key metrics
- Insights for next phases

**How to Use**:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

### 4. Unit Tests ([tests/unit/test_ingestion.py](tests/unit/test_ingestion.py))

**Purpose**: Test ingestion components

**Tests Include**:
- Download URL generation
- Schema validation
- DataFrame preparation
- Statistics computation
- Database insertion (mocked)

**How to Run**:
```bash
# Run ingestion tests only
pytest tests/unit/test_ingestion.py -v

# Run all unit tests
make test-unit

# Run with coverage
pytest tests/unit/test_ingestion.py -v --cov=src.ingestion
```

---

### 5. Quick Start Script ([scripts/quickstart_ingestion.sh](scripts/quickstart_ingestion.sh))

**Purpose**: Interactive script for easy ingestion

**Features**:
- Checks MySQL connection
- Initializes database if needed
- Interactive menu for different ingestion options
- Automatic verification

**How to Use**:
```bash
./scripts/quickstart_ingestion.sh
```

**Menu Options**:
1. Sample data (10K rows) - Recommended for testing
2. Full month (~3M rows)
3. Multiple months (Jan-Mar)
4. Test download only
5. Run unit tests

---

## Data Source

**NYC Taxi & Limousine Commission (TLC) Trip Record Data**

- **URL**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Direct Download**: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
- **Format**: Parquet (compressed, efficient)
- **Size**: ~40-50MB per month (compressed), ~3M rows per month
- **Authentication**: None required (public data)
- **Update Frequency**: Monthly

**Dataset Features**:
- Pickup/dropoff datetime
- Trip distance
- Pickup/dropoff locations
- Passenger count
- Fare amount, tips, tolls
- Payment type
- And more...

---

## Quick Start Guide

### Option 1: Using Makefile (Easiest)

```bash
# Start MySQL
docker-compose up -d mysql

# Ingest sample data
make ingest

# View data
make db-shell
# Then: SELECT COUNT(*) FROM raw_taxi_trips;
```

### Option 2: Using Quick Start Script

```bash
# Run interactive script
./scripts/quickstart_ingestion.sh

# Follow menu prompts
```

### Option 3: Manual Commands

```bash
# 1. Ensure MySQL is running
docker-compose up -d mysql

# 2. Download and ingest (sample)
python -m src.ingestion.ingest_data \
    --year 2023 \
    --start-month 1 \
    --end-month 1 \
    --sample 10000 \
    --mode replace

# 3. Verify in MySQL
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "SELECT COUNT(*) FROM raw_taxi_trips"

# 4. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Database Schema

Data is inserted into the `raw_taxi_trips` table:

```sql
CREATE TABLE raw_taxi_trips (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT,
    pickup_datetime DATETIME,
    dropoff_datetime DATETIME,
    passenger_count INT,
    trip_distance FLOAT,
    pickup_location_id INT,
    dropoff_location_id INT,
    rate_code INT,
    store_and_fwd_flag VARCHAR(1),
    payment_type INT,
    fare_amount FLOAT,
    extra FLOAT,
    mta_tax FLOAT,
    tip_amount FLOAT,
    tolls_amount FLOAT,
    total_amount FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pickup_datetime (pickup_datetime),
    INDEX idx_payment_type (payment_type)
);
```

---

## Performance Notes

**Sample Data (10K rows)**:
- Download: ~5 seconds
- Ingestion: ~2 seconds
- Total: ~7 seconds
- Use case: Testing and development

**Full Month (~3M rows)**:
- Download: ~30-60 seconds (depends on connection)
- Ingestion: ~30-60 seconds
- Total: ~1-2 minutes
- Use case: Real analysis and training

**Multiple Months (3 months)**:
- Download: ~2-3 minutes
- Ingestion: ~3-5 minutes
- Total: ~5-8 minutes
- Use case: Production ML pipeline

---

## Verification Commands

```bash
# Check row count
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "SELECT COUNT(*) as total_rows FROM raw_taxi_trips"

# View date range
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "SELECT MIN(pickup_datetime), MAX(pickup_datetime) FROM raw_taxi_trips"

# Sample data
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "SELECT * FROM raw_taxi_trips LIMIT 5"

# Payment type distribution
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "SELECT payment_type, COUNT(*) as count FROM raw_taxi_trips GROUP BY payment_type"
```

---

## Common Issues & Solutions

### Issue 1: MySQL Connection Error

**Error**: `Can't connect to MySQL server`

**Solutions**:
```bash
# Start MySQL with Docker
docker-compose up -d mysql

# Wait for MySQL to be ready
sleep 10

# Check if running
docker-compose ps

# Check logs
docker-compose logs mysql
```

### Issue 2: Download Fails

**Error**: `Download failed: Connection error`

**Solutions**:
- Check internet connection
- Try alternative month (some months may not be available)
- Download manually from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### Issue 3: Permission Denied on Scripts

**Error**: `Permission denied: ./scripts/quickstart_ingestion.sh`

**Solution**:
```bash
chmod +x scripts/quickstart_ingestion.sh
```

### Issue 4: Database Already Exists Error

**Error**: `Table 'raw_taxi_trips' already exists`

**Solutions**:
```bash
# Use append mode instead of replace
python -m src.ingestion.ingest_data --mode append ...

# Or drop table first
mysql -h localhost -u smartanalytics -p smartanalytics_db \
    -e "DROP TABLE IF EXISTS raw_taxi_trips"

# Then run init script
mysql -h localhost -u smartanalytics -p smartanalytics_db < scripts/init_db.sql
```

---

## What's Next: Phase 2 - Data Validation & Cleaning

Now that data is ingested, the next step is to:

1. **Validate data quality**
   - Schema validation
   - Missing value analysis
   - Outlier detection
   - Duplicate detection

2. **Clean the data**
   - Handle missing values
   - Remove/cap outliers
   - Fix data types
   - Remove invalid records

3. **Generate quality reports**
   - Summary statistics
   - Quality metrics
   - Visualizations
   - Data drift detection

**Files to create in Phase 2**:
- `src/processing/validate_data.py`
- `src/processing/clean_data.py`
- `src/processing/quality_report.py`
- `notebooks/02_data_quality.ipynb`
- `tests/unit/test_processing.py`

**Command to run Phase 2** (once implemented):
```bash
make process
```

---

## Summary

✅ **Completed**:
- Data download from NYC TLC
- Database ingestion pipeline
- Schema validation
- Data exploration notebook
- Unit tests
- Quick start scripts
- Documentation

✅ **Database**:
- Raw data loaded into MySQL
- Indexed for performance
- Ready for processing

✅ **Next Steps**:
- Phase 2: Data validation and cleaning
- Phase 3: Feature engineering
- Phase 4: Model training

**Total Time**: Phase 1 implementation took ~2 hours

**Data Available**: NYC Yellow Taxi trips (Jan 2023 onwards)

🚀 **Ready for Phase 2!**
