import gevent.monkey
gevent.monkey.patch_all()

import pandas as pd
from cassandra.io.geventreactor import GeventConnection
from cassandra.cluster import Cluster

CSV_PATH = "SkyNest_India_Combined_Flight_Dataset.csv"

# Start with 1,000 rows. Increase to None after verifying.
LIMIT_ROWS = None

cluster = Cluster(
    ["127.0.0.1"],
    port=9042,
    connection_class=GeventConnection
)
session = cluster.connect("skynest")

insert = session.prepare("""
    INSERT INTO flights_by_id (
        flight_id, flight_date, airline, flight_number,
        origin_iata, origin_city, destination_iata,
        destination_city, scheduled_departure,
        scheduled_arrival, scheduled_duration_min,
        weather_condition, temperature_c, visibility_km,
        wind_speed_kmh, precipitation_mm,
        airport_traffic_level, public_holiday_flag,
        departure_delay_min, delay_target, delay_category,
        data_type
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

def get_value(row, column):
    value = row.get(column)
    if pd.isna(value):
        return None
    return value

def get_text(row, column):
    value = get_value(row, column)
    return None if value is None else str(value)

def get_int(row, column):
    value = get_value(row, column)
    return None if value is None else int(float(value))

def get_float(row, column):
    value = get_value(row, column)
    return None if value is None else float(value)

def get_date(row, column):
    value = get_value(row, column)
    if value is None:
        return None
    return pd.to_datetime(value).date()

def get_bool(row, column):
    value = get_value(row, column)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")

try:
    df = pd.read_csv(CSV_PATH, nrows=LIMIT_ROWS)
    df.columns = df.columns.str.strip()

    required = [
        "flight_id", "flight_date", "airline",
        "origin_iata", "destination_iata",
        "departure_delay_min", "delay_target"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    inserted = 0

    for _, row in df.iterrows():
        flight_id = get_text(row, "flight_id")
        if not flight_id:
            continue

        session.execute(insert, (
            flight_id,
            get_date(row, "flight_date"),
            get_text(row, "airline"),
            get_text(row, "flight_number"),
            get_text(row, "origin_iata"),
            get_text(row, "origin_city"),
            get_text(row, "destination_iata"),
            get_text(row, "destination_city"),
            get_text(row, "scheduled_departure"),
            get_text(row, "scheduled_arrival"),
            get_int(row, "scheduled_duration_min"),
            get_text(row, "weather_condition"),
            get_float(row, "temperature_c"),
            get_float(row, "visibility_km"),
            get_float(row, "wind_speed_kmh"),
            get_float(row, "precipitation_mm"),
            get_text(row, "airport_traffic_level"),
            get_bool(row, "public_holiday_flag"),
            get_int(row, "departure_delay_min"),
            get_int(row, "delay_target"),
            get_text(row, "delay_category"),
            get_text(row, "data_type")
        ))

        inserted += 1
        if inserted % 100 == 0:
            print(f"Imported {inserted} rows")

    print(f"Import complete: {inserted} rows")

finally:
    cluster.shutdown()
