"""Quick script to verify data ingestion."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.database import db

print("=" * 60)
print("DATA INGESTION VERIFICATION")
print("=" * 60)

# 1. Row count
result = db.execute_query("SELECT COUNT(*) FROM raw_taxi_trips")
total_rows = result[0][0]
print(f"\n✓ Total rows in database: {total_rows:,}")

# 2. Date range
result = db.execute_query("""
    SELECT
        MIN(pickup_datetime) as earliest,
        MAX(pickup_datetime) as latest
    FROM raw_taxi_trips
""")
print(f"✓ Date range: {result[0][0]} to {result[0][1]}")

# 3. Sample data
print("\n" + "=" * 60)
print("SAMPLE DATA (first 5 rows)")
print("=" * 60)
df = db.read_table("raw_taxi_trips", limit=5)
print(df.to_string())

# 4. Basic statistics
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
stats = db.execute_query("""
    SELECT
        COUNT(*) as total_trips,
        AVG(trip_distance) as avg_distance,
        AVG(fare_amount) as avg_fare,
        MIN(fare_amount) as min_fare,
        MAX(fare_amount) as max_fare
    FROM raw_taxi_trips
""")
print(f"Total trips: {stats[0][0]:,}")
print(f"Average distance: {stats[0][1]:.2f} miles")
print(f"Average fare: ${stats[0][2]:.2f}")
print(f"Min fare: ${stats[0][3]:.2f}")
print(f"Max fare: ${stats[0][4]:.2f}")

# 5. Payment type distribution
print("\n" + "=" * 60)
print("PAYMENT TYPE DISTRIBUTION")
print("=" * 60)
payment_dist = db.execute_query("""
    SELECT
        payment_type,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_taxi_trips), 2) as percentage
    FROM raw_taxi_trips
    GROUP BY payment_type
    ORDER BY count DESC
""")
print("Payment Type | Count | Percentage")
print("-" * 40)
for row in payment_dist:
    print(f"Type {row[0]:8} | {row[1]:5,} | {row[2]:6.2f}%")

print("\n" + "=" * 60)
print("✓ VERIFICATION COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
print("  2. Start Phase 2: Data cleaning and validation")
print("  3. Ingest more data: make ingest-full")
