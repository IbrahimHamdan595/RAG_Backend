from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor, Json

import config


@contextmanager
def _db():
    conn = psycopg2.connect(config.SUPABASE_DB_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _to_pg(v):
    if isinstance(v, (dict, list)):
        return Json(v)
    return v


def _as_dict(row):
    return dict(row) if row else None


class Collection:
    def __init__(self, table):
        self._table = table

    def _where(self, f):
        conds, vals = [], []
        for k, v in (f or {}).items():
            if isinstance(v, dict) and "$in" in v:
                conds.append(f"{k} = ANY(%s)")
                vals.append(v["$in"])
            else:
                conds.append(f"{k} = %s")
                vals.append(v)
        return (" AND ".join(conds) if conds else "TRUE"), vals

    def find_one(self, f):
        where, vals = self._where(f)
        with _db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {self._table} WHERE {where} LIMIT 1", vals)
                return _as_dict(cur.fetchone())

    def find(self, f=None):
        where, vals = self._where(f)
        with _db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {self._table} WHERE {where}", vals)
                return [_as_dict(r) for r in cur.fetchall()]

    def insert_one(self, doc):
        cols = list(doc.keys())
        sql = (
            f"INSERT INTO {self._table} ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))})"
        )
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [_to_pg(doc[c]) for c in cols])

    def insert_many(self, docs):
        if not docs:
            return
        cols = list(docs[0].keys())
        sql = (
            f"INSERT INTO {self._table} ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))})"
        )
        with _db() as conn:
            with conn.cursor() as cur:
                for doc in docs:
                    cur.execute(sql, [_to_pg(doc.get(c)) for c in cols])

    def update_one(self, f, update):
        sets = update.get("$set", {})
        if not sets:
            return
        where, where_vals = self._where(f)
        set_sql = ", ".join(f"{k} = %s" for k in sets)
        set_vals = [_to_pg(v) for v in sets.values()]
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self._table} SET {set_sql} WHERE {where}",
                    set_vals + where_vals,
                )

    def count_documents(self, f=None):
        where, vals = self._where(f)
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table} WHERE {where}", vals)
                return cur.fetchone()[0]


documents_collection = Collection("documents")
units_collection     = Collection("units")
chunks_collection    = Collection("chunks")


def create_document_record(file_name, file_type, storage_path):
    doc = {
        "document_id":  str(uuid4()),
        "file_name":    file_name,
        "file_type":    file_type,
        "uploaded_at":  datetime.utcnow(),
        "status":       "uploaded",
        "storage_path": storage_path,
        "total_units":  None,
    }
    documents_collection.insert_one(doc)
    return doc


def create_unit_record(document_id, unit_number, raw_text, unit_type="page",
                       title=None, clean_text=None, metadata=None):
    unit = {
        "unit_id":     str(uuid4()),
        "document_id": document_id,
        "unit_type":   unit_type,
        "unit_number": unit_number,
        "title":       title,
        "raw_text":    raw_text,
        "clean_text":  clean_text,
        "metadata":    metadata or {"unit_number": unit_number, "unit_type": unit_type},
        "created_at":  datetime.utcnow(),
    }
    units_collection.insert_one(unit)
    return unit


def create_chunk_record(document_id, unit_id, chunk_index, chunk_text,
                        token_count, overlap_from_previous=False):
    chunk = {
        "chunk_id":    str(uuid4()),
        "document_id": document_id,
        "unit_id":     unit_id,
        "chunk_index": chunk_index,
        "text":        chunk_text,
        "metadata":    {"overlap_from_previous": overlap_from_previous},
        "created_at":  datetime.utcnow(),
    }
    chunks_collection.insert_one(chunk)
    return chunk


def update_document_units(document_id, total_units):
    documents_collection.update_one(
        {"document_id": document_id},
        {"$set": {"total_units": total_units, "status": "processed"}},
    )
