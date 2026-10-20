
import gevent.monkey
gevent.monkey.patch_all()

import pandas as pd
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.io.geventreactor import GeventConnection


@st.cache_resource
def get_cassandra_session():
    cluster = Cluster(
        ["127.0.0.1"],
        port=9042,
        connection_class=GeventConnection
    )
    session = cluster.connect("skynest")
    return cluster, session


def get_sample_flights(limit=100):
    cluster, session = get_cassandra_session()

    rows = session.execute(
        """
        SELECT flight_id, airline, origin_iata,
               destination_iata, departure_delay_min, delay_target
        FROM flights_by_id
        LIMIT %s
        """,
        (limit,)
    )

    return pd.DataFrame([
        {
            "flight_id": r.flight_id,
            "airline": r.airline,
            "origin_iata": r.origin_iata,
            "destination_iata": r.destination_iata,
            "departure_delay_min": r.departure_delay_min,
            "delay_target": r.delay_target
        }
        for r in rows
    ])
