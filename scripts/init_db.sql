-- Smart Analytics Database Initialization

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS smartanalytics_db;
USE smartanalytics_db;

-- Raw data tables
CREATE TABLE IF NOT EXISTS raw_taxi_trips (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT,
    pickup_datetime DATETIME,
    dropoff_datetime DATETIME,
    passenger_count INT,
    trip_distance FLOAT,
    pickup_longitude FLOAT,
    pickup_latitude FLOAT,
    rate_code INT,
    store_and_fwd_flag VARCHAR(1),
    dropoff_longitude FLOAT,
    dropoff_latitude FLOAT,
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Processed data tables
CREATE TABLE IF NOT EXISTS processed_taxi_trips (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT,
    pickup_datetime DATETIME,
    dropoff_datetime DATETIME,
    passenger_count INT,
    trip_distance FLOAT,
    pickup_longitude FLOAT,
    pickup_latitude FLOAT,
    dropoff_longitude FLOAT,
    dropoff_latitude FLOAT,
    payment_type INT,
    fare_amount FLOAT,
    tip_amount FLOAT,
    total_amount FLOAT,
    trip_duration INT,
    trip_duration_minutes FLOAT,
    speed_mph FLOAT,
    is_weekend BOOLEAN,
    hour_of_day INT,
    day_of_week INT,
    month INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pickup_datetime (pickup_datetime),
    INDEX idx_payment_type (payment_type),
    INDEX idx_trip_duration (trip_duration)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Feature tables
CREATE TABLE IF NOT EXISTS feature_store (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    trip_id BIGINT,
    feature_name VARCHAR(100),
    feature_value FLOAT,
    feature_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trip_id (trip_id),
    INDEX idx_feature_name (feature_name),
    INDEX idx_feature_version (feature_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Data quality tracking
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    threshold_value FLOAT,
    status VARCHAR(20),
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_name (table_name),
    INDEX idx_checked_at (checked_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Model metadata
CREATE TABLE IF NOT EXISTS model_registry (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100),
    model_type VARCHAR(50),
    model_version VARCHAR(20),
    model_path VARCHAR(500),
    mlflow_run_id VARCHAR(100),
    metrics JSON,
    parameters JSON,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_model_name (model_name),
    INDEX idx_model_type (model_type),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Prediction logs
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_id BIGINT,
    input_features JSON,
    prediction FLOAT,
    prediction_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_id (model_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (model_id) REFERENCES model_registry(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Experiment tracking
CREATE TABLE IF NOT EXISTS experiment_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100),
    run_id VARCHAR(100) UNIQUE,
    run_name VARCHAR(200),
    parameters JSON,
    metrics JSON,
    tags JSON,
    status VARCHAR(20),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_name (experiment_name),
    INDEX idx_run_id (run_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create views for common queries
CREATE OR REPLACE VIEW v_trip_summary AS
SELECT
    DATE(pickup_datetime) AS trip_date,
    COUNT(*) AS total_trips,
    AVG(trip_distance) AS avg_distance,
    AVG(trip_duration_minutes) AS avg_duration,
    AVG(fare_amount) AS avg_fare,
    AVG(tip_amount) AS avg_tip,
    SUM(total_amount) AS total_revenue
FROM processed_taxi_trips
GROUP BY DATE(pickup_datetime);

CREATE OR REPLACE VIEW v_payment_distribution AS
SELECT
    payment_type,
    COUNT(*) AS trip_count,
    AVG(tip_amount) AS avg_tip,
    AVG(total_amount) AS avg_total
FROM processed_taxi_trips
GROUP BY payment_type;

-- Grant permissions
GRANT ALL PRIVILEGES ON smartanalytics_db.* TO 'smartanalytics'@'%';
FLUSH PRIVILEGES;
