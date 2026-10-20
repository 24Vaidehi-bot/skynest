
import gevent.monkey
gevent.monkey.patch_all()

import pandas as pd
from cassandra.cluster import Cluster
from cassandra.io.geventreactor import GeventConnection

cluster = None

try:
    cluster = Cluster(
        ["127.0.0.1"],
        port=9042,
        connection_class=GeventConnection
    )

    session = cluster.connect("skynest")

    rows = session.execute("""
        SELECT flight_id, airline, origin_iata,
               destination_iata, departure_delay_min, delay_target
        FROM flights_by_id
        LIMIT 10
    """)

    df = pd.DataFrame(
        [
            {
                "flight_id": r.flight_id,
                "airline": r.airline,
                "origin_iata": r.origin_iata,
                "destination_iata": r.destination_iata,
                "departure_delay_min": r.departure_delay_min,
                "delay_target": r.delay_target
            }
            for r in rows
        ]
    )

    print("SkyNest Cassandra connection successful.")
    print("Records retrieved:", len(df))
    print(df.to_string(index=False))

finally:
    if cluster is not None:
        cluster.shutdown()
