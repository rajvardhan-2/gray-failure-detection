"""
Read Router for Gray-Failure Detection and Self-Healing
in a Replicated PostgreSQL Database.

Expected health report fields:
    replica_id: str
    status: str
    response_time_ms: float
    replication_lag_bytes: int
    checked_at: float (Unix timestamp in seconds)

Requires:
    pip install "psycopg[binary]>=3.1"
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    import psycopg
    from psycopg import InterfaceError, OperationalError

except ImportError:
    psycopg = None

    class OperationalError(Exception):
        pass

    class InterfaceError(Exception):
        pass


class ReadRouterError(RuntimeError):
    """Raised when the router cannot complete a read."""


class ReadRouter:

    def __init__(
        self,
        replica_dsns: Mapping[str, str],
        health_reports_function: Callable,
        primary_dsn_provider: Callable,
        *,
        max_response_time_ms: float = 200.0,
        max_replication_lag_bytes: int = 1_000_000,
        max_report_age_seconds: float = 5.0,
        connect_timeout_seconds: int = 3,
        statement_timeout_ms: int = 3000,
    ):

        if not replica_dsns:
            raise ValueError("At least one replica DSN is required")

        if not callable(health_reports_function):
            raise TypeError("health_reports_function must be callable")

        if not callable(primary_dsn_provider):
            raise TypeError("primary_dsn_provider must be callable")

        if max_response_time_ms < 0 or max_replication_lag_bytes < 0:
            raise ValueError("Eligibility thresholds cannot be negative")

        if max_report_age_seconds < 0:
            raise ValueError("Report age cannot be negative")

        if connect_timeout_seconds <= 0 or statement_timeout_ms <= 0:
            raise ValueError("Timeout values must be positive")

        self.replica_dsns = dict(replica_dsns)
        self.health_reports_function = health_reports_function
        self.primary_dsn_provider = primary_dsn_provider

        self.max_response_time_ms = max_response_time_ms
        self.max_replication_lag_bytes = max_replication_lag_bytes
        self.max_report_age_seconds = max_report_age_seconds

        self.connect_timeout_seconds = connect_timeout_seconds
        self.statement_timeout_ms = statement_timeout_ms

    # ---------------------------------------------------------
    # 1. CHECK REPLICA ELIGIBILITY
    # ---------------------------------------------------------

    def is_eligible(
        self,
        report: Mapping[str, Any],
        now: float | None = None
    ) -> bool:

        if not isinstance(report, Mapping):
            return False

        current_time = time.time() if now is None else now

        try:
            replica_id = report["replica_id"]
            status = report["status"]
            response_time = float(report["response_time_ms"])
            lag_bytes = float(report["replication_lag_bytes"])
            checked_at = float(report["checked_at"])

        except (KeyError, TypeError, ValueError, OverflowError):
            return False

        if not isinstance(replica_id, str):
            return False

        if replica_id not in self.replica_dsns:
            return False

        if not isinstance(status, str):
            return False

        if status.strip().lower() != "healthy":
            return False

        values = (
            response_time,
            lag_bytes,
            checked_at,
            current_time
        )

        if not all(math.isfinite(value) for value in values):
            return False

        if response_time < 0 or lag_bytes < 0:
            return False

        if response_time > self.max_response_time_ms:
            return False

        if lag_bytes > self.max_replication_lag_bytes:
            return False

        report_age = current_time - checked_at

        # Reject stale and future-dated health reports.
        return 0 <= report_age <= self.max_report_age_seconds

    # ---------------------------------------------------------
    # 2. GET ELIGIBLE REPLICAS
    # ---------------------------------------------------------

    def get_eligible_replicas(
        self,
        reports: Sequence[Mapping[str, Any]]
    ) -> list[str]:

        eligible = []
        seen = set()
        now = time.time()

        for report in reports:

            if not self.is_eligible(report, now):
                continue

            replica_id = report["replica_id"]

            # Ignore duplicate reports for the same replica.
            if replica_id in seen:
                continue

            seen.add(replica_id)

            response_time = float(report["response_time_ms"])

            eligible.append((response_time, replica_id))

        # Fastest replica first.
        eligible.sort(key=lambda item: (item[0], item[1]))

        return [replica_id for _, replica_id in eligible]

    # ---------------------------------------------------------
    # 3. EXECUTE READ QUERY
    # ---------------------------------------------------------

    def execute_read(
        self,
        query: str,
        params: Any = None
    ) -> list[tuple[Any, ...]]:

        self._validate_read_query(query)

        try:
            reports = self.health_reports_function()

        except Exception as exc:
            raise ReadRouterError(
                f"Could not obtain health reports: {exc}"
            ) from exc

        if reports is None:
            reports = []

        connection_errors = []

        # First try eligible replicas.
        eligible_replicas = self.get_eligible_replicas(reports)

        for replica_id in eligible_replicas:

            dsn = self.replica_dsns.get(replica_id)

            if not dsn:
                continue

            try:
                return self._run_query(dsn, query, params)

            except (OperationalError, InterfaceError) as exc:

                connection_errors.append(
                    f"{replica_id}: {exc}"
                )

                # Try the next eligible replica.

        # If replicas fail, resolve the current primary.
        # This supports Patroni leader changes.

        try:
            primary_dsn = self.primary_dsn_provider()

            if not isinstance(primary_dsn, str) or not primary_dsn.strip():
                raise ValueError("Primary DSN provider returned an empty DSN")

        except Exception as exc:

            details = "; ".join(connection_errors)

            raise ReadRouterError(
                f"Could not resolve current primary. "
                f"Replica errors: {details}. "
                f"Primary resolution error: {exc}"
            ) from exc

        # Final fallback: current primary.
        try:
            return self._run_query(primary_dsn, query, params)

        except (OperationalError, InterfaceError) as exc:

            connection_errors.append(
                f"Current primary: {exc}"
            )

            raise ReadRouterError(
                "Read failed on all permitted destinations. "
                + "; ".join(connection_errors)
            ) from exc

    # ---------------------------------------------------------
    # 4. RUN QUERY ON A DATABASE NODE
    # ---------------------------------------------------------

    def _run_query(
        self,
        dsn: str,
        query: str,
        params: Any = None
    ) -> list[tuple[Any, ...]]:

        if psycopg is None:
            raise RuntimeError(
                'Install the PostgreSQL driver using: '
                'pip install "psycopg[binary]>=3.1"'
            )

        with psycopg.connect(
            dsn,
            connect_timeout=self.connect_timeout_seconds,
            application_name="gray_failure_read_router"
        ) as connection:

            with connection.transaction():

                # Enforce read-only transaction.
                connection.execute(
                    "SET TRANSACTION READ ONLY"
                )

                # Prevent indefinitely running queries.
                connection.execute(
                    "SELECT set_config("
                    "'statement_timeout', %s, true)",
                    (str(self.statement_timeout_ms),)
                )

                cursor = connection.execute(query, params)

                if cursor.description is None:
                    raise ReadRouterError(
                        "The supplied SQL did not return a result set"
                    )

                return cursor.fetchall()

    # ---------------------------------------------------------
    # 5. BASIC QUERY VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_read_query(query: str):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty SQL string"
            )


# -------------------------------------------------------------
# 6. CREATE ROUTER USING ENVIRONMENT VARIABLES
# -------------------------------------------------------------

def create_router(
    health_reports_function: Callable,
    primary_dsn_provider: Callable
) -> ReadRouter:

    replica_dsns = {
        "replica_1": os.getenv("REPLICA_1_DSN", "").strip(),
        "replica_2": os.getenv("REPLICA_2_DSN", "").strip()
    }

    replica_dsns = {
        key: value
        for key, value in replica_dsns.items()
        if value
    }

    if not replica_dsns:
        raise ValueError(
            "Set at least one replica DSN environment variable"
        )

    return ReadRouter(
        replica_dsns=replica_dsns,
        health_reports_function=health_reports_function,
        primary_dsn_provider=primary_dsn_provider
    )