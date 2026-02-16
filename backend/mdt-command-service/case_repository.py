"""
Voxelomics MDT Command Service - Case Persistence Backends

Provides repository implementations for case storage:
- SqliteCaseRepository (offline/runtime default)
- FirestoreCaseRepository (cloud runtime backend)
- InMemoryCaseRepository (test fallback)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from models import MDTCaseRecord

try:
    from google.cloud import firestore
except Exception:
    firestore = None


class CaseRepository:
    def save_case(self, record: MDTCaseRecord) -> MDTCaseRecord:
        raise NotImplementedError

    def get_case(self, case_id: str) -> MDTCaseRecord:
        raise NotImplementedError

    def list_patient_case_history(
        self,
        patient_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_error: bool = False,
    ) -> List[dict]:
        raise NotImplementedError

    def get_case_history_snapshot(self, snapshot_id: int) -> tuple[datetime, MDTCaseRecord]:
        raise NotImplementedError

    def delete_case_history_snapshot(self, snapshot_id: int) -> bool:
        raise NotImplementedError


class InMemoryCaseRepository(CaseRepository):
    def __init__(self) -> None:
        self._store: Dict[str, MDTCaseRecord] = {}
        self._history: List[dict] = []
        self._next_snapshot_id = 1

    def save_case(self, record: MDTCaseRecord) -> MDTCaseRecord:
        self._store[record.case_id] = record
        if record.status.value in {"pending_approval", "approved", "rework_required", "error"}:
            now = datetime.now(timezone.utc)
            self._history.append(
                {
                    "snapshot_id": self._next_snapshot_id,
                    "case_id": record.case_id,
                    "patient_id": record.patient_id,
                    "patient_name": record.patient_name,
                    "diagnosis": record.diagnosis,
                    "status": record.status.value,
                    "saved_at": now,
                    "updated_at": record.updated_at,
                    "record": record,
                }
            )
            self._next_snapshot_id += 1
        return record

    def get_case(self, case_id: str) -> MDTCaseRecord:
        record = self._store.get(case_id)
        if not record:
            raise KeyError(f"Case not found: {case_id}. Start the case first via /mdt/start.")
        return record

    def list_patient_case_history(
        self,
        patient_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_error: bool = False,
    ) -> List[dict]:
        rows = [x for x in self._history if x["patient_id"] == patient_id]
        if since is not None:
            rows = [x for x in rows if x["saved_at"] >= since]
        if not include_error:
            rows = [x for x in rows if x["status"] != "error"]
        rows.sort(key=lambda x: x["saved_at"], reverse=True)
        return rows[: max(1, limit)]

    def get_case_history_snapshot(self, snapshot_id: int) -> tuple[datetime, MDTCaseRecord]:
        for row in self._history:
            if row["snapshot_id"] == snapshot_id:
                return row["saved_at"], row["record"]
        raise KeyError(f"Case history snapshot not found: {snapshot_id}.")

    def delete_case_history_snapshot(self, snapshot_id: int) -> bool:
        snapshot_id = int(snapshot_id)
        for idx, row in enumerate(self._history):
            if int(row["snapshot_id"]) == snapshot_id:
                del self._history[idx]
                return True
        return False


class SqliteCaseRepository(CaseRepository):
    def __init__(self, db_path: str) -> None:
        if not db_path:
            raise RuntimeError("SQLite repository requires a non-empty db_path.")
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mdt_cases (
                    case_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mdt_case_history (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    diagnosis TEXT NOT NULL,
                    status TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mdt_case_history_patient_saved
                ON mdt_case_history(patient_id, saved_at DESC)
                """
            )
            conn.commit()

    def save_case(self, record: MDTCaseRecord) -> MDTCaseRecord:
        payload = record.model_dump_json()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mdt_cases(case_id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(case_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (record.case_id, payload),
            )
            if record.status.value in {"pending_approval", "approved", "rework_required", "error"}:
                saved_at = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    INSERT INTO mdt_case_history (
                        case_id,
                        patient_id,
                        patient_name,
                        diagnosis,
                        status,
                        saved_at,
                        updated_at,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.case_id,
                        record.patient_id,
                        record.patient_name,
                        record.diagnosis,
                        record.status.value,
                        saved_at,
                        record.updated_at.isoformat(),
                        payload,
                    ),
                )
            conn.commit()
        return record

    def get_case(self, case_id: str) -> MDTCaseRecord:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM mdt_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Case not found: {case_id}. Start the case first via /mdt/start.")
        return MDTCaseRecord.model_validate_json(row["payload_json"])

    def list_patient_case_history(
        self,
        patient_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_error: bool = False,
    ) -> List[dict]:
        sql = """
            SELECT snapshot_id, case_id, patient_id, patient_name, diagnosis, status, saved_at, updated_at
            FROM mdt_case_history
            WHERE patient_id = ?
        """
        params: List[object] = [patient_id]
        if since is not None:
            sql += " AND saved_at >= ?"
            params.append(since.isoformat())
        if not include_error:
            sql += " AND status != ?"
            params.append("error")
        sql += " ORDER BY saved_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        out: List[dict] = []
        for row in rows:
            out.append(
                {
                    "snapshot_id": int(row["snapshot_id"]),
                    "case_id": str(row["case_id"]),
                    "patient_id": str(row["patient_id"]),
                    "patient_name": str(row["patient_name"]),
                    "diagnosis": str(row["diagnosis"]),
                    "status": str(row["status"]),
                    "saved_at": datetime.fromisoformat(str(row["saved_at"])),
                    "updated_at": datetime.fromisoformat(str(row["updated_at"])),
                }
            )
        return out

    def get_case_history_snapshot(self, snapshot_id: int) -> tuple[datetime, MDTCaseRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT saved_at, payload_json
                FROM mdt_case_history
                WHERE snapshot_id = ?
                """,
                (int(snapshot_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Case history snapshot not found: {snapshot_id}.")
        saved_at = datetime.fromisoformat(str(row["saved_at"]))
        record = MDTCaseRecord.model_validate_json(str(row["payload_json"]))
        return saved_at, record

    def delete_case_history_snapshot(self, snapshot_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM mdt_case_history WHERE snapshot_id = ?",
                (int(snapshot_id),),
            )
            conn.commit()
            return bool(cursor.rowcount and cursor.rowcount > 0)


class FirestoreCaseRepository(CaseRepository):
    def __init__(self, project_id: str, collection: str = "mdt_cases") -> None:
        if firestore is None:
            raise RuntimeError(
                "google-cloud-firestore is unavailable. Install requirements.txt first."
            )
        if not project_id:
            raise RuntimeError("Firestore repository requires GOOGLE_CLOUD_PROJECT.")
        self.project_id = project_id
        self.collection_name = collection
        self.client = firestore.Client(project=project_id)
        self.collection = self.client.collection(collection)

    def save_case(self, record: MDTCaseRecord) -> MDTCaseRecord:
        payload = record.model_dump(mode="json")
        self.collection.document(record.case_id).set(payload)
        return record

    def get_case(self, case_id: str) -> MDTCaseRecord:
        doc = self.collection.document(case_id).get()
        if not doc.exists:
            raise KeyError(f"Case not found: {case_id}. Start the case first via /mdt/start.")
        payload = doc.to_dict() or {}
        return MDTCaseRecord.model_validate(payload)

    def list_patient_case_history(
        self,
        patient_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_error: bool = False,
    ) -> List[dict]:
        raise NotImplementedError("Case history listing is not implemented for Firestore backend.")

    def get_case_history_snapshot(self, snapshot_id: int) -> tuple[datetime, MDTCaseRecord]:
        raise NotImplementedError("Case history snapshots are not implemented for Firestore backend.")

    def delete_case_history_snapshot(self, snapshot_id: int) -> bool:
        raise NotImplementedError("Case history snapshots are not implemented for Firestore backend.")
