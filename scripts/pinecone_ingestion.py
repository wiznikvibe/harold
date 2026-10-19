"""Incremental Pinecone ingestion for Harold logistics corpus."""

from __future__ import annotations

import argparse
import sys
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


logger = logging.getLogger("pinecone_ingestion")

EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEFAULT_INDEX_NAME = "harold-logistics"
POLICY_DIR_DEFAULT = "datasource/policy"
TABLE_NAME = "dynamic_supply_chain_logistics"
STATE_TABLE = "ingestion_document_state"
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".csv", ".xlsx"}
DEFAULT_ROW_BATCH_SIZE = 50


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RowWorkItem:
    document_id: str
    content_hash: str
    chunks: list[Document]
    state: dict | None
    old_version: int | None
    old_chunk_count: int
    new_version: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def vector_id(document_id: str, version: int, chunk_index: int) -> str:
    return f"{document_id}-v{version}-c{chunk_index}"


def previous_vector_ids(document_id: str, version: int, chunk_count: int) -> list[str]:
    return [vector_id(document_id, version, i) for i in range(chunk_count)]


def row_document_id(table: str, row_id: int | str) -> str:
    return f"{table}-{row_id}"


def canonical_row_payload(row: dict[str, Any]) -> str:
    # Sort keys; convert non-JSON types via default=str
    ordered = {k: row[k] for k in sorted(row.keys())}
    return json.dumps(ordered, ensure_ascii=False, sort_keys=True, default=str)


def row_to_text(row: dict[str, Any]) -> str:
    lines = []
    for key in sorted(row.keys()):
        val = row[key]
        if val is None:
            continue
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


def discover_policy_files(policy_dir: str) -> list[Path]:
    root = Path(policy_dir)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def parse_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    source = path.name
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                text,
                {
                    "source_file": source,
                    "file_format": suffix.lstrip("."),
                    "document_type": "policy",
                    "page_number": 1,
                },
            )
        ]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            docs.append(
                Document(
                    page.extract_text() or "",
                    {
                        "source_file": source,
                        "file_format": "pdf",
                        "document_type": "policy",
                        "page_number": i,
                    },
                )
            )
        return docs
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        docs = []
        for idx, row in df.iterrows():
            payload = row.dropna().to_dict()
            docs.append(
                Document(
                    row_to_text(payload),
                    {
                        "source_file": source,
                        "file_format": "csv",
                        "document_type": "tabular_file",
                        "row_index": int(idx),
                    },
                )
            )
        return docs
    if suffix == ".xlsx":
        import pandas as pd

        df = pd.read_excel(path)
        docs = []
        for idx, row in df.iterrows():
            payload = row.dropna().to_dict()
            docs.append(
                Document(
                    row_to_text(payload),
                    {
                        "source_file": source,
                        "file_format": "xlsx",
                        "document_type": "tabular_file",
                        "row_index": int(idx),
                    },
                )
            )
        return docs
    raise ValueError(f"Unsupported file type: {path}")


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    out: list[Document] = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        for chunk in splitter.split_text(doc.page_content):
            out.append(Document(chunk, dict(doc.metadata)))
    return out


def fetch_table_rows(conn, limit: int | None = None) -> list[dict]:
    sql = f"SELECT * FROM {TABLE_NAME}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def row_as_document(row: dict) -> Document:
    row_id = row["id"]
    return Document(
        row_to_text(row),
        {
            "source_file": TABLE_NAME,
            "file_format": "mysql_row",
            "document_type": "telemetry",
            "row_id": row_id,
            "row_index": row_id,
        },
    )


def load_config() -> dict[str, str]:
    load_dotenv()
    cfg = {
        "pinecone_api_key": os.environ["PINECONE_API_KEY"],
        "hf_api_key": os.environ.get("HUGGING_FACE_API_KEY", ""),
        "mysql_host": os.environ["MYSQL_HOST"],
        "mysql_port": int(os.environ.get("MYSQL_PORT", "3306")),
        "mysql_user": os.environ["MYSQL_USER"],
        "mysql_password": os.environ["MYSQL_PASSWORD"],
        "mysql_database": os.environ["MYSQL_DATABASE"],
        "index_name": os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL),
        "policy_dir": os.environ.get("POLICY_DIR", POLICY_DIR_DEFAULT),
    }
    if cfg["embedding_model"] != EMBEDDING_MODEL:
        raise ValueError(
            f"Only {EMBEDDING_MODEL} is supported; got {cfg['embedding_model']}"
        )
    return cfg


def get_mysql_connection(cfg: dict | None = None):
    cfg = cfg or load_config()
    return pymysql.connect(
        host=cfg["mysql_host"],
        port=cfg["mysql_port"],
        user=cfg["mysql_user"],
        password=cfg["mysql_password"],
        database=cfg["mysql_database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def state_schema_statements() -> list[str]:
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
            document_id VARCHAR(512) NOT NULL PRIMARY KEY,
            content_hash CHAR(64) NOT NULL,
            version INT NOT NULL,
            chunk_count INT NOT NULL DEFAULT 0,
            status VARCHAR(16) NOT NULL,
            embedding_model VARCHAR(128) NOT NULL,
            source_type VARCHAR(32) NOT NULL,
            last_error TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    ]


def init_state_schema(conn) -> None:
    with conn.cursor() as cur:
        for stmt in state_schema_statements():
            cur.execute(stmt)


def fetch_document_state(conn, document_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT * FROM {STATE_TABLE} WHERE document_id = %s",
            (document_id,),
        )
        return cur.fetchone()


def fetch_document_states(conn, document_ids: list[str]) -> dict[str, dict]:
    """Bulk-fetch state rows keyed by document_id."""
    if not document_ids:
        return {}
    out: dict[str, dict] = {}
    # Stay under typical max_allowed_packet / placeholder limits.
    chunk = 500
    with conn.cursor() as cur:
        for i in range(0, len(document_ids), chunk):
            batch = document_ids[i : i + chunk]
            placeholders = ",".join(["%s"] * len(batch))
            cur.execute(
                f"SELECT * FROM {STATE_TABLE} WHERE document_id IN ({placeholders})",
                batch,
            )
            for row in cur.fetchall():
                out[row["document_id"]] = row
    return out


def upsert_document_state(
    conn,
    *,
    document_id: str,
    content_hash: str,
    version: int,
    chunk_count: int,
    status: str,
    embedding_model: str,
    source_type: str,
    last_error: str | None = None,
) -> None:
    sql = f"""
    INSERT INTO {STATE_TABLE}
      (document_id, content_hash, version, chunk_count, status,
       embedding_model, source_type, last_error)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      content_hash=VALUES(content_hash),
      version=VALUES(version),
      chunk_count=VALUES(chunk_count),
      status=VALUES(status),
      embedding_model=VALUES(embedding_model),
      source_type=VALUES(source_type),
      last_error=VALUES(last_error)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                document_id,
                content_hash,
                version,
                chunk_count,
                status,
                embedding_model,
                source_type,
                last_error,
            ),
        )


def delete_document_state(conn, document_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {STATE_TABLE} WHERE document_id = %s",
            (document_id,),
        )


def list_document_ids(conn, source_type: str | None = None) -> list[str]:
    with conn.cursor() as cur:
        if source_type:
            cur.execute(
                f"SELECT document_id FROM {STATE_TABLE} WHERE source_type = %s",
                (source_type,),
            )
        else:
            cur.execute(f"SELECT document_id FROM {STATE_TABLE}")
        return [r["document_id"] for r in cur.fetchall()]


def _ensure_hf_hub_token() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_API_KEY")
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def build_embeddings(model_name: str = EMBEDDING_MODEL):
    _ensure_hf_hub_token()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings) -> int:
    vec = embeddings.embed_query("dimension probe")
    return len(vec)


def ensure_pinecone_index(
    pc: Pinecone, index_name: str, dimension: int, metric: str = "cosine"
):
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        desc = pc.describe_index(index_name)
        existing_dim = desc.dimension
        if existing_dim != dimension:
            raise RuntimeError(
                f"Pinecone index '{index_name}' dimension {existing_dim} "
                f"!= embedding dimension {dimension}. Refusing to delete/recreate."
            )
    return pc.Index(index_name)


def _chunk_metadata(
    document_id: str,
    version: int,
    chunk_index: int,
    chunk: Document,
    ingested_at: str,
) -> dict:
    meta = {
        "document_id": document_id,
        "source_file": chunk.metadata.get("source_file", document_id),
        "version": version,
        "chunk_index": chunk_index,
        "embedding_model": EMBEDDING_MODEL,
        "file_format": chunk.metadata.get("file_format", ""),
        "document_type": chunk.metadata.get("document_type", ""),
        "ingested_at": ingested_at,
    }
    for k in ("page_number", "row_index", "row_id"):
        if k in chunk.metadata and chunk.metadata[k] is not None:
            meta[k] = chunk.metadata[k]
    return {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in meta.items()
    }


def upsert_vectors_payload(
    index,
    vectors_payload: list[dict],
    batch_size: int = 50,
    max_retries: int = 3,
    cleanup_label: str = "batch",
) -> int:
    """Upsert prepared vectors in batches; on failure delete any IDs already written."""
    upserted = 0
    committed_ids: list[str] = []
    try:
        for i in range(0, len(vectors_payload), batch_size):
            batch = vectors_payload[i : i + batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    index.upsert(vectors=batch)
                    upserted += len(batch)
                    committed_ids.extend(v["id"] for v in batch)
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        raise
                    logger.warning("upsert retry %s/%s: %s", attempt, max_retries, exc)
                    time.sleep(2**attempt)
    except Exception:
        if committed_ids:
            try:
                delete_vectors_by_ids(index, committed_ids)
            except Exception:
                logger.exception("cleanup failed after partial upsert (%s)", cleanup_label)
        raise
    return upserted


def upsert_chunks(
    index,
    embeddings,
    document_id,
    version,
    chunks,
    batch_size=50,
    max_retries=3,
) -> int:
    texts = [c.page_content for c in chunks]
    embedded: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        embedded.extend(embeddings.embed_documents(texts[i : i + batch_size]))
    ingested_at = datetime.now(timezone.utc).isoformat()
    vectors_payload = []
    for i, (chunk, values) in enumerate(zip(chunks, embedded)):
        vectors_payload.append(
            {
                "id": vector_id(document_id, version, i),
                "values": values,
                "metadata": _chunk_metadata(document_id, version, i, chunk, ingested_at),
            }
        )
    return upsert_vectors_payload(
        index,
        vectors_payload,
        batch_size=batch_size,
        max_retries=max_retries,
        cleanup_label=f"{document_id} v{version}",
    )


def delete_vectors_by_ids(index, ids: list[str], batch_size: int = 100) -> None:
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        if batch:
            index.delete(ids=batch)


def failure_state_values(
    prior: dict | None,
    *,
    new_content_hash: str = "",
) -> tuple[str, int, int, str]:
    """Fields to preserve on failure when no successful write occurred."""
    if prior:
        return (
            prior["content_hash"],
            prior["version"],
            prior["chunk_count"],
            prior["embedding_model"],
        )
    return (new_content_hash, 0, 0, EMBEDDING_MODEL)


def upsert_outer_failure_state(
    conn,
    document_id: str,
    *,
    source_type: str,
    last_error: str,
    new_content_hash: str = "",
) -> None:
    prior = fetch_document_state(conn, document_id)
    content_hash, version, chunk_count, embedding_model = failure_state_values(
        prior, new_content_hash=new_content_hash
    )
    upsert_document_state(
        conn,
        document_id=document_id,
        content_hash=content_hash,
        version=version,
        chunk_count=chunk_count,
        status="failed",
        embedding_model=embedding_model,
        source_type=source_type,
        last_error=last_error[:1000],
    )


def process_document(
    *,
    conn,
    index,
    embeddings,
    document_id: str,
    content_hash: str,
    source_type: str,
    chunks: list[Document],
) -> str:
    """Returns 'skipped' | 'success' | 'failed'."""
    state = fetch_document_state(conn, document_id)
    if state and state["content_hash"] == content_hash and state["status"] == "success":
        return "skipped"

    old_version = state["version"] if state else None
    old_chunk_count = state["chunk_count"] if state else 0
    new_version = (old_version or 0) + 1

    try:
        if not chunks:
            raise ValueError("no chunks produced")
        upserted = upsert_chunks(index, embeddings, document_id, new_version, chunks)
        if upserted != len(chunks):
            raise RuntimeError(f"upserted {upserted} != chunks {len(chunks)}")
        if old_version is not None and old_chunk_count:
            delete_vectors_by_ids(
                index, previous_vector_ids(document_id, old_version, old_chunk_count)
            )
        upsert_document_state(
            conn,
            document_id=document_id,
            content_hash=content_hash,
            version=new_version,
            chunk_count=len(chunks),
            status="success",
            embedding_model=EMBEDDING_MODEL,
            source_type=source_type,
            last_error=None,
        )
        return "success"
    except Exception as exc:
        logger.exception("failed processing %s", document_id)
        if chunks:
            try:
                delete_vectors_by_ids(
                    index,
                    previous_vector_ids(document_id, new_version, len(chunks)),
                )
            except Exception:
                logger.exception(
                    "cleanup new-version vectors after failure for %s v%s",
                    document_id,
                    new_version,
                )
        ch, ver, cc, em = failure_state_values(state, new_content_hash=content_hash)
        upsert_document_state(
            conn,
            document_id=document_id,
            content_hash=ch,
            version=ver,
            chunk_count=cc,
            status="failed",
            embedding_model=em,
            source_type=source_type,
            last_error=str(exc)[:1000],
        )
        return "failed"


def plan_row_work_items(conn, rows: list[dict]) -> tuple[int, list[RowWorkItem]]:
    """Classify rows into skip vs work. Returns (skipped_count, items_to_process)."""
    doc_ids = [row_document_id(TABLE_NAME, row["id"]) for row in rows]
    states = fetch_document_states(conn, doc_ids)
    skipped = 0
    items: list[RowWorkItem] = []
    for row in rows:
        doc_id = row_document_id(TABLE_NAME, row["id"])
        content_hash = sha256_text(canonical_row_payload(row))
        state = states.get(doc_id)
        if state and state["content_hash"] == content_hash and state["status"] == "success":
            skipped += 1
            continue
        old_version = state["version"] if state else None
        old_chunk_count = state["chunk_count"] if state else 0
        items.append(
            RowWorkItem(
                document_id=doc_id,
                content_hash=content_hash,
                chunks=[row_as_document(row)],
                state=state,
                old_version=old_version,
                old_chunk_count=old_chunk_count or 0,
                new_version=(old_version or 0) + 1,
            )
        )
    return skipped, items


def _mark_row_items_failed(
    conn,
    index,
    items: list[RowWorkItem],
    *,
    scrub_new_version: bool,
    last_error: str,
) -> None:
    for item in items:
        if scrub_new_version and item.chunks:
            try:
                delete_vectors_by_ids(
                    index,
                    previous_vector_ids(
                        item.document_id, item.new_version, len(item.chunks)
                    ),
                )
            except Exception:
                logger.exception(
                    "cleanup new-version vectors after batch failure for %s v%s",
                    item.document_id,
                    item.new_version,
                )
        ch, ver, cc, em = failure_state_values(
            item.state, new_content_hash=item.content_hash
        )
        upsert_document_state(
            conn,
            document_id=item.document_id,
            content_hash=ch,
            version=ver,
            chunk_count=cc,
            status="failed",
            embedding_model=em,
            source_type="table_row",
            last_error=last_error[:1000],
        )


def process_row_batch(
    *,
    conn,
    index,
    embeddings,
    items: list[RowWorkItem],
    batch_size: int = DEFAULT_ROW_BATCH_SIZE,
    max_retries: int = 3,
) -> list[str]:
    """
    Embed + upsert a batch of changed rows together.
    Returns a status per item in the same order: 'success' | 'failed'.
    """
    if not items:
        return []

    statuses: dict[str, str] = {}
    active: list[RowWorkItem] = []
    for item in items:
        if not item.chunks:
            _mark_row_items_failed(
                conn,
                index,
                [item],
                scrub_new_version=False,
                last_error="no chunks produced",
            )
            statuses[item.document_id] = "failed"
        else:
            active.append(item)

    if not active:
        return [statuses[item.document_id] for item in items]

    texts: list[str] = []
    owners: list[tuple[RowWorkItem, int]] = []
    for item in active:
        for i, chunk in enumerate(item.chunks):
            texts.append(chunk.page_content)
            owners.append((item, i))

    try:
        embedded: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            embedded.extend(embeddings.embed_documents(texts[i : i + batch_size]))
        if len(embedded) != len(texts):
            raise RuntimeError(
                f"embedded {len(embedded)} vectors != texts {len(texts)}"
            )

        ingested_at = datetime.now(timezone.utc).isoformat()
        vectors_payload: list[dict] = []
        for (item, chunk_index), values in zip(owners, embedded):
            chunk = item.chunks[chunk_index]
            vectors_payload.append(
                {
                    "id": vector_id(item.document_id, item.new_version, chunk_index),
                    "values": values,
                    "metadata": _chunk_metadata(
                        item.document_id,
                        item.new_version,
                        chunk_index,
                        chunk,
                        ingested_at,
                    ),
                }
            )

        upserted = upsert_vectors_payload(
            index,
            vectors_payload,
            batch_size=batch_size,
            max_retries=max_retries,
            cleanup_label=f"row-batch:{len(active)}",
        )
        if upserted != len(vectors_payload):
            raise RuntimeError(
                f"upserted {upserted} != vectors {len(vectors_payload)}"
            )
    except Exception as exc:
        logger.exception("row batch embed/upsert failed (%d rows)", len(active))
        _mark_row_items_failed(
            conn,
            index,
            active,
            scrub_new_version=True,
            last_error=str(exc),
        )
        for item in active:
            statuses[item.document_id] = "failed"
        return [statuses[item.document_id] for item in items]

    for item in active:
        try:
            if item.old_version is not None and item.old_chunk_count:
                delete_vectors_by_ids(
                    index,
                    previous_vector_ids(
                        item.document_id, item.old_version, item.old_chunk_count
                    ),
                )
            upsert_document_state(
                conn,
                document_id=item.document_id,
                content_hash=item.content_hash,
                version=item.new_version,
                chunk_count=len(item.chunks),
                status="success",
                embedding_model=EMBEDDING_MODEL,
                source_type="table_row",
                last_error=None,
            )
            statuses[item.document_id] = "success"
        except Exception as exc:
            logger.exception("post-upsert finalize failed for %s", item.document_id)
            _mark_row_items_failed(
                conn,
                index,
                [item],
                scrub_new_version=True,
                last_error=str(exc),
            )
            statuses[item.document_id] = "failed"

    return [statuses[item.document_id] for item in items]


def process_changed_rows(
    *,
    conn,
    index,
    embeddings,
    rows: list[dict],
    row_batch_size: int = DEFAULT_ROW_BATCH_SIZE,
) -> tuple[int, int, int]:
    """
    Plan and process rows in batches.
    Returns (skipped, success, failed).
    """
    skipped, items = plan_row_work_items(conn, rows)
    success = 0
    failed = 0
    if skipped:
        logger.info("rows skipped (unchanged): %d", skipped)
    for i in range(0, len(items), row_batch_size):
        batch = items[i : i + row_batch_size]
        logger.info(
            "processing row batch %d-%d of %d changed",
            i + 1,
            i + len(batch),
            len(items),
        )
        try:
            statuses = process_row_batch(
                conn=conn,
                index=index,
                embeddings=embeddings,
                items=batch,
                batch_size=row_batch_size,
            )
            for item, status in zip(batch, statuses):
                logger.info("row %s -> %s", item.document_id, status)
                if status == "success":
                    success += 1
                else:
                    failed += 1
        except Exception as exc:
            failed += len(batch)
            logger.exception("outer failure row batch: %s", exc)
            for item in batch:
                upsert_outer_failure_state(
                    conn,
                    item.document_id,
                    source_type="table_row",
                    last_error=str(exc),
                    new_content_hash=item.content_hash,
                )
    return skipped, success, failed


def should_process_files(
    policy_dir: str | Path,
    discovered_ids: set[str],
    tracked_file_ids: set[str],
) -> tuple[bool, str]:
    """Whether file ingestion may reconcile and process. False => do not call reconcile_deletions."""
    if not Path(policy_dir).is_dir():
        return False, "policy_dir_not_directory"
    if not discovered_ids and tracked_file_ids:
        return False, "empty_discovery_with_tracked"
    return True, ""


def reconcile_deletions(conn, index, current_ids: set[str], source_type: str) -> int:
    deleted = 0
    for doc_id in list_document_ids(conn, source_type=source_type):
        if doc_id in current_ids:
            continue
        state = fetch_document_state(conn, doc_id)
        if state and state.get("chunk_count"):
            delete_vectors_by_ids(
                index,
                previous_vector_ids(doc_id, state["version"], state["chunk_count"]),
            )
        delete_document_state(conn, doc_id)
        deleted += 1
    return deleted


def run_ingestion(
    files_only=False,
    rows_only=False,
    limit_rows=None,
    policy_dir=None,
    row_batch_size: int = DEFAULT_ROW_BATCH_SIZE,
) -> int:
    """Run ingestion. Returns 0 on full success, 1 if any failure or skipped file pass."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    policy_dir = policy_dir or cfg["policy_dir"]
    conn = get_mysql_connection(cfg)
    init_state_schema(conn)
    embeddings = build_embeddings(cfg["embedding_model"])
    dim = embedding_dimension(embeddings)
    pc = Pinecone(api_key=cfg["pinecone_api_key"])
    index = ensure_pinecone_index(pc, cfg["index_name"], dim)
    if row_batch_size < 1:
        raise ValueError("row_batch_size must be >= 1")

    failures = 0

    if not rows_only:
        tracked_file_ids = set(list_document_ids(conn, source_type="file"))
        policy_path = Path(policy_dir)
        files = discover_policy_files(policy_dir) if policy_path.is_dir() else []
        discovered_ids = {p.name for p in files}
        should_process, skip_reason = should_process_files(
            policy_dir, discovered_ids, tracked_file_ids
        )
        if not should_process:
            failures += 1
            if skip_reason == "policy_dir_not_directory":
                logger.error(
                    "policy_dir missing or not a directory: %s; skipping file ingestion",
                    policy_dir,
                )
            elif skip_reason == "empty_discovery_with_tracked":
                logger.error(
                    "policy_dir %s yielded no files but %d tracked file documents exist; "
                    "skipping file reconcile and processing",
                    policy_dir,
                    len(tracked_file_ids),
                )
        else:
            reconcile_deletions(conn, index, discovered_ids, "file")
            for path in files:
                content_hash = ""
                try:
                    raw = path.read_bytes()
                    content_hash = sha256_bytes(raw)
                    chunks = chunk_documents(parse_file(path))
                    status = process_document(
                        conn=conn,
                        index=index,
                        embeddings=embeddings,
                        document_id=path.name,
                        content_hash=content_hash,
                        source_type="file",
                        chunks=chunks,
                    )
                    logger.info("file %s -> %s", path.name, status)
                    if status == "failed":
                        failures += 1
                except Exception as exc:
                    failures += 1
                    logger.exception("outer failure file %s: %s", path.name, exc)
                    upsert_outer_failure_state(
                        conn,
                        path.name,
                        source_type="file",
                        last_error=str(exc),
                        new_content_hash=content_hash,
                    )

    if not files_only:
        rows = fetch_table_rows(conn, limit=limit_rows)
        current_row_ids = {row_document_id(TABLE_NAME, r["id"]) for r in rows}
        if limit_rows is None:
            reconcile_deletions(conn, index, current_row_ids, "table_row")
        try:
            _skipped, _ok, row_failed = process_changed_rows(
                conn=conn,
                index=index,
                embeddings=embeddings,
                rows=rows,
                row_batch_size=row_batch_size,
            )
            failures += row_failed
        except Exception as exc:
            failures += 1
            logger.exception("outer failure processing rows: %s", exc)

    conn.close()
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Harold Pinecone incremental ingestion")
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--rows-only", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=DEFAULT_ROW_BATCH_SIZE,
        help=f"Rows to embed/upsert together (default {DEFAULT_ROW_BATCH_SIZE})",
    )
    parser.add_argument("--policy-dir", default=None)
    args = parser.parse_args()
    sys.exit(
        run_ingestion(
            files_only=args.files_only,
            rows_only=args.rows_only,
            limit_rows=args.limit_rows,
            policy_dir=args.policy_dir,
            row_batch_size=args.row_batch_size,
        )
    )


if __name__ == "__main__":
    main()
