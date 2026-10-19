"""Incremental Pinecone ingestion for Harold logistics corpus."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEFAULT_INDEX_NAME = "harold-logistics"
POLICY_DIR_DEFAULT = "datasource/policy"
TABLE_NAME = "dynamic_supply_chain_logistics"
STATE_TABLE = "ingestion_document_state"
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".csv", ".xlsx"}


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


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
