# Pinecone Incremental Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one runnable script that incrementally ingests `datasource/policy/` files and MySQL `dynamic_supply_chain_logistics` rows into Pinecone index `harold-logistics` with MySQL state tracking and upsert-then-delete safety.

**Architecture:** Single module `scripts/pinecone_ingestion.py` owns config, MySQL state, parsers, embed/upsert, and the main loop. Pure helpers are unit-tested; integration is exercised via CLI with `--limit-rows` and files-only flags.

**Tech Stack:** Python 3.10+, `python-dotenv`, `pymysql`, `pinecone`, `langchain-huggingface` / `sentence-transformers`, `langchain-text-splitters`, `pypdf`, `pandas`/`openpyxl` for xlsx/csv.

**Spec:** `docs/superpowers/specs/2026-10-19-pinecone-ingestion-design.md`

## Global Constraints

- Embedding model: exactly `google/embeddinggemma-300m` (no dual modes).
- State store: MySQL table `ingestion_document_state` (never JSON file).
- Update order: upsert new → verify count → delete old → mark success.
- Vector IDs: `{document_id}-v{version}-c{chunk_index}`.
- Never auto-delete a Pinecone index on dimension mismatch; raise instead.
- Secrets only from `.env` / environment — never hardcode passwords in source.
- Default index name: `harold-logistics`.
- Policy files path: `datasource/policy/`.
- Table: `logistics.dynamic_supply_chain_logistics`, one document per row.

---

## File map

| Path | Responsibility |
|------|----------------|
| `scripts/pinecone_ingestion.py` | Full pipeline (create/fill) |
| `scripts/requirements-ingest.txt` | Pinned-ish deps for the script |
| `tests/test_ingestion_helpers.py` | Unit tests for hash, IDs, row text, state SQL helpers (mocked) |
| `.env` | Add MySQL connection vars (modify; do not commit secrets) |

---

### Task 1: Dependencies, env keys, and pure helpers + tests

**Files:**
- Create: `scripts/requirements-ingest.txt`
- Create: `tests/test_ingestion_helpers.py`
- Create: `scripts/pinecone_ingestion.py` (helpers only in this task)
- Modify: `.env` (add MySQL keys; do not commit)

**Interfaces:**
- Produces:
  - `sha256_bytes(data: bytes) -> str`
  - `sha256_text(text: str) -> str`
  - `vector_id(document_id: str, version: int, chunk_index: int) -> str`
  - `previous_vector_ids(document_id: str, version: int, chunk_count: int) -> list[str]`
  - `row_to_text(row: dict) -> str`
  - `row_document_id(table: str, row_id: int | str) -> str`
  - `canonical_row_payload(row: dict) -> str` (stable JSON for hashing)

- [ ] **Step 1: Write failing tests for pure helpers**

Create `tests/test_ingestion_helpers.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pinecone_ingestion as ing


def test_vector_id_format():
    assert ing.vector_id("SOP.md", 2, 3) == "SOP.md-v2-c3"


def test_previous_vector_ids():
    ids = ing.previous_vector_ids("doc", 1, 3)
    assert ids == ["doc-v1-c0", "doc-v1-c1", "doc-v1-c2"]


def test_row_document_id():
    assert (
        ing.row_document_id("dynamic_supply_chain_logistics", 42)
        == "dynamic_supply_chain_logistics-42"
    )


def test_canonical_row_payload_stable():
    a = ing.canonical_row_payload({"b": 1, "a": None})
    b = ing.canonical_row_payload({"a": None, "b": 1})
    assert a == b
    assert json.loads(a)["a"] is None


def test_sha256_bytes_deterministic():
    assert ing.sha256_bytes(b"abc") == ing.sha256_bytes(b"abc")
    assert ing.sha256_bytes(b"abc") != ing.sha256_bytes(b"abd")
```

- [ ] **Step 2: Run tests — expect ImportError / AttributeError**

Run: `cd /home/kai/repo/harold && python -m pytest tests/test_ingestion_helpers.py -v`  
Expected: FAIL (module missing helpers)

- [ ] **Step 3: Add requirements file**

Create `scripts/requirements-ingest.txt`:

```
python-dotenv>=1.0.0
pymysql>=1.1.0
pinecone>=5.0.0
langchain-huggingface>=0.1.0
langchain-text-splitters>=0.3.0
sentence-transformers>=3.0.0
pypdf>=4.0.0
pandas>=2.0.0
openpyxl>=3.1.0
pytest>=8.0.0
```

Install: `pip install -r scripts/requirements-ingest.txt`

- [ ] **Step 4: Implement pure helpers in `scripts/pinecone_ingestion.py`**

```python
"""Incremental Pinecone ingestion for Harold logistics corpus."""

from __future__ import annotations

import hashlib
import json
from typing import Any


EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEFAULT_INDEX_NAME = "harold-logistics"
POLICY_DIR_DEFAULT = "datasource/policy"
TABLE_NAME = "dynamic_supply_chain_logistics"
STATE_TABLE = "ingestion_document_state"


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
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `python -m pytest tests/test_ingestion_helpers.py -v`  
Expected: all PASS

- [ ] **Step 6: Add MySQL keys to `.env` (local only)**

Append (values from operator; do not commit `.env`):

```
MYSQL_HOST=100.31.187.36
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=<from operator>
MYSQL_DATABASE=logistics
PINECONE_INDEX_NAME=harold-logistics
EMBEDDING_MODEL=google/embeddinggemma-300m
```

- [ ] **Step 7: Commit**

```bash
git add scripts/requirements-ingest.txt scripts/pinecone_ingestion.py tests/test_ingestion_helpers.py docs/superpowers/specs/2026-10-19-pinecone-ingestion-design.md docs/superpowers/plans/2026-10-19-pinecone-ingestion.md
git commit -m "$(cat <<'EOF'
feat: add ingestion helper foundations and design/plan docs

EOF
)"
```

---

### Task 2: MySQL state schema and helpers

**Files:**
- Modify: `scripts/pinecone_ingestion.py`
- Modify: `tests/test_ingestion_helpers.py`

**Interfaces:**
- Consumes: constants `STATE_TABLE`
- Produces:
  - `get_mysql_connection() -> pymysql.Connection`
  - `init_state_schema(conn) -> None`
  - `fetch_document_state(conn, document_id: str) -> dict | None`
  - `upsert_document_state(conn, *, document_id, content_hash, version, chunk_count, status, embedding_model, source_type, last_error=None) -> None`
  - `delete_document_state(conn, document_id: str) -> None`
  - `list_document_ids(conn, source_type: str | None = None) -> list[str]`

- [ ] **Step 1: Add failing tests with mocked cursor (or skip live DB)**

Append to `tests/test_ingestion_helpers.py`:

```python
def test_init_state_schema_sql_contains_table_name():
    statements = ing.state_schema_statements()
    assert any(ing.STATE_TABLE in s for s in statements)
    assert any("content_hash" in s for s in statements)
    assert any("chunk_count" in s for s in statements)
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python -m pytest tests/test_ingestion_helpers.py::test_init_state_schema_sql_contains_table_name -v`  
Expected: FAIL (`state_schema_statements` missing)

- [ ] **Step 3: Implement MySQL helpers**

Add to `scripts/pinecone_ingestion.py`:

```python
import os
from datetime import datetime, timezone

import pymysql
from dotenv import load_dotenv


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
```

- [ ] **Step 4: Live smoke — create table**

Run:

```bash
cd /home/kai/repo/harold && python - <<'PY'
from scripts import pinecone_ingestion as ing
# if scripts is not a package, import via path:
import sys
sys.path.insert(0, 'scripts')
import pinecone_ingestion as ing
conn = ing.get_mysql_connection()
ing.init_state_schema(conn)
print('OK schema')
conn.close()
PY
```

Expected: `OK schema` and table visible in MySQL.

- [ ] **Step 5: Run unit tests — PASS**

Run: `python -m pytest tests/test_ingestion_helpers.py -v`

- [ ] **Step 6: Commit**

```bash
git add scripts/pinecone_ingestion.py tests/test_ingestion_helpers.py
git commit -m "$(cat <<'EOF'
feat: add MySQL ingestion state schema helpers

EOF
)"
```

---

### Task 3: Embedding model + Pinecone index create/validate

**Files:**
- Modify: `scripts/pinecone_ingestion.py`

**Interfaces:**
- Produces:
  - `build_embeddings(model_name: str)` → embedding object with `.embed_documents(list[str]) -> list[list[float]]`
  - `embedding_dimension(embeddings) -> int`
  - `ensure_pinecone_index(pc, index_name: str, dimension: int, metric: str = "cosine") -> Index`
  - Raises `RuntimeError` if existing index dimension ≠ model dimension

- [ ] **Step 1: Implement embedding + index helpers**

```python
from pinecone import Pinecone, ServerlessSpec


def build_embeddings(model_name: str = EMBEDDING_MODEL):
    # Prefer local sentence-transformers load of the HF model
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings) -> int:
    vec = embeddings.embed_query("dimension probe")
    return len(vec)


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int, metric: str = "cosine"):
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
```

- [ ] **Step 2: Smoke test init (may download model; long)**

```bash
cd /home/kai/repo/harold && python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
import pinecone_ingestion as ing
from pinecone import Pinecone
cfg = ing.load_config()
emb = ing.build_embeddings(cfg['embedding_model'])
dim = ing.embedding_dimension(emb)
print('dim', dim)
pc = Pinecone(api_key=cfg['pinecone_api_key'])
index = ing.ensure_pinecone_index(pc, cfg['index_name'], dim)
print('index ok', cfg['index_name'])
PY
```

Expected: prints dimension (likely 768) and `index ok harold-logistics`.

- [ ] **Step 3: Commit**

```bash
git add scripts/pinecone_ingestion.py
git commit -m "$(cat <<'EOF'
feat: initialize EmbeddingGemma and validate Pinecone index

EOF
)"
```

---

### Task 4: File + row parsers and chunking

**Files:**
- Modify: `scripts/pinecone_ingestion.py`
- Modify: `tests/test_ingestion_helpers.py`

**Interfaces:**
- Produces:
  - `discover_policy_files(policy_dir: str) -> list[Path]`
  - `parse_file(path: Path) -> list[DocumentLike]` where DocumentLike has `.page_content` and `.metadata`
  - `chunk_documents(docs, chunk_size=800, chunk_overlap=120) -> list[DocumentLike]`
  - `fetch_table_rows(conn, limit: int | None) -> list[dict]`
  - `row_as_document(row: dict) -> DocumentLike`

- [ ] **Step 1: Add test for row_to_text skipping nulls**

```python
def test_row_to_text_skips_nulls():
    text = ing.row_to_text({"id": 1, "RISK_CLS_TXT": "HIGH", "x": None})
    assert "id: 1" in text
    assert "RISK_CLS_TXT: HIGH" in text
    assert "x:" not in text
```

- [ ] **Step 2: Implement parsers**

```python
from pathlib import Path
from dataclasses import dataclass, field


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".csv", ".xlsx"}


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


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
        return [Document(text, {"source_file": source, "file_format": suffix.lstrip("."), "document_type": "policy", "page_number": 1})]
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            docs.append(Document(page.extract_text() or "", {
                "source_file": source, "file_format": "pdf", "document_type": "policy", "page_number": i
            }))
        return docs
    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        docs = []
        for idx, row in df.iterrows():
            payload = row.dropna().to_dict()
            docs.append(Document(row_to_text(payload), {
                "source_file": source, "file_format": "csv", "document_type": "tabular_file", "row_index": int(idx)
            }))
        return docs
    if suffix == ".xlsx":
        import pandas as pd
        df = pd.read_excel(path)
        docs = []
        for idx, row in df.iterrows():
            payload = row.dropna().to_dict()
            docs.append(Document(row_to_text(payload), {
                "source_file": source, "file_format": "xlsx", "document_type": "tabular_file", "row_index": int(idx)
            }))
        return docs
    raise ValueError(f"Unsupported file type: {path}")


def chunk_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
```

- [ ] **Step 3: Smoke parse policy files**

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
import pinecone_ingestion as ing
files = ing.discover_policy_files('datasource/policy')
assert files, 'no files'
for f in files:
    docs = ing.parse_file(f)
    chunks = ing.chunk_documents(docs)
    print(f.name, 'docs', len(docs), 'chunks', len(chunks))
PY
```

Expected: prints `SOP.md` and `policy.md` with chunk counts > 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/pinecone_ingestion.py tests/test_ingestion_helpers.py
git commit -m "$(cat <<'EOF'
feat: parse policy files and MySQL telemetry rows

EOF
)"
```

---

### Task 5: Embed + upsert with retries and ID construction

**Files:**
- Modify: `scripts/pinecone_ingestion.py`
- Modify: `tests/test_ingestion_helpers.py`

**Interfaces:**
- Produces:
  - `upsert_chunks(index, embeddings, document_id, version, chunks: list[Document], batch_size=50, max_retries=3) -> int`
  - `delete_vectors_by_ids(index, ids: list[str], batch_size=100) -> None`

- [ ] **Step 1: Test ID list length equals chunk_count helper (already covered); add upsert batching unit test with fake index**

```python
class FakeIndex:
    def __init__(self):
        self.upserts = []
        self.deletes = []
    def upsert(self, vectors):
        self.upserts.extend(vectors)
        return {"upserted_count": len(vectors)}
    def delete(self, ids):
        self.deletes.extend(ids)


class FakeEmb:
    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]


def test_upsert_chunks_builds_hierarchical_ids():
    idx = FakeIndex()
    chunks = [ing.Document("a", {}), ing.Document("b", {})]
    n = ing.upsert_chunks(idx, FakeEmb(), "SOP.md", 3, chunks, batch_size=10)
    assert n == 2
    assert idx.upserts[0]["id"] == "SOP.md-v3-c0"
    assert idx.upserts[1]["id"] == "SOP.md-v3-c1"
```

- [ ] **Step 2: Implement upsert/delete**

```python
import time
import logging

logger = logging.getLogger("pinecone_ingestion")


def upsert_chunks(index, embeddings, document_id, version, chunks, batch_size=50, max_retries=3) -> int:
    texts = [c.page_content for c in chunks]
    vectors_payload = []
    # Embed in batches too
    embedded = []
    for i in range(0, len(texts), batch_size):
        embedded.extend(embeddings.embed_documents(texts[i:i + batch_size]))
    ingested_at = datetime.now(timezone.utc).isoformat()
    for i, (chunk, values) in enumerate(zip(chunks, embedded)):
        meta = {
            "document_id": document_id,
            "source_file": chunk.metadata.get("source_file", document_id),
            "version": version,
            "chunk_index": i,
            "embedding_model": EMBEDDING_MODEL,
            "file_format": chunk.metadata.get("file_format", ""),
            "document_type": chunk.metadata.get("document_type", ""),
            "ingested_at": ingested_at,
        }
        for k in ("page_number", "row_index", "row_id"):
            if k in chunk.metadata and chunk.metadata[k] is not None:
                meta[k] = chunk.metadata[k]
        # Pinecone metadata values must be primitives
        clean = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}
        vectors_payload.append({"id": vector_id(document_id, version, i), "values": values, "metadata": clean})

    upserted = 0
    for i in range(0, len(vectors_payload), batch_size):
        batch = vectors_payload[i:i + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                index.upsert(vectors=batch)
                upserted += len(batch)
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise
                logger.warning("upsert retry %s/%s: %s", attempt, max_retries, exc)
                time.sleep(2 ** attempt)
    return upserted


def delete_vectors_by_ids(index, ids: list[str], batch_size: int = 100) -> None:
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        if batch:
            index.delete(ids=batch)
```

- [ ] **Step 3: Run unit tests — PASS**

Run: `python -m pytest tests/test_ingestion_helpers.py -v`

- [ ] **Step 4: Commit**

```bash
git add scripts/pinecone_ingestion.py tests/test_ingestion_helpers.py
git commit -m "$(cat <<'EOF'
feat: upsert vectors with hierarchical IDs and retries

EOF
)"
```

---

### Task 6: Per-document process + main loop (upsert-then-delete)

**Files:**
- Modify: `scripts/pinecone_ingestion.py`

**Interfaces:**
- Produces:
  - `process_document(...)` implementing skip / version bump / upsert / verify / delete old / state update
  - `reconcile_deletions(...)`
  - `main()` with argparse: `--limit-rows`, `--files-only`, `--rows-only`

- [ ] **Step 1: Implement process_document and reconcile**

```python
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
        # Preserve prior successful version fields if present; still mark failed
        upsert_document_state(
            conn,
            document_id=document_id,
            content_hash=state["content_hash"] if state else content_hash,
            version=state["version"] if state else 0,
            chunk_count=state["chunk_count"] if state else 0,
            status="failed",
            embedding_model=state["embedding_model"] if state else EMBEDDING_MODEL,
            source_type=source_type,
            last_error=str(exc)[:1000],
        )
        return "failed"


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
```

- [ ] **Step 2: Implement `main()`**

```python
import argparse


def run_ingestion(files_only=False, rows_only=False, limit_rows=None, policy_dir=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    policy_dir = policy_dir or cfg["policy_dir"]
    conn = get_mysql_connection(cfg)
    init_state_schema(conn)
    embeddings = build_embeddings(cfg["embedding_model"])
    dim = embedding_dimension(embeddings)
    pc = Pinecone(api_key=cfg["pinecone_api_key"])
    index = ensure_pinecone_index(pc, cfg["index_name"], dim)

    if not rows_only:
        files = discover_policy_files(policy_dir)
        current_file_ids = {p.name for p in files}
        reconcile_deletions(conn, index, current_file_ids, "file")
        for path in files:
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
            except Exception as exc:
                logger.exception("outer failure file %s: %s", path.name, exc)
                upsert_document_state(
                    conn,
                    document_id=path.name,
                    content_hash="",
                    version=0,
                    chunk_count=0,
                    status="failed",
                    embedding_model=EMBEDDING_MODEL,
                    source_type="file",
                    last_error=str(exc)[:1000],
                )

    if not files_only:
        rows = fetch_table_rows(conn, limit=limit_rows)
        current_row_ids = {
            row_document_id(TABLE_NAME, r["id"]) for r in rows
        }
        # Only reconcile full-table deletions when not limiting
        if limit_rows is None:
            reconcile_deletions(conn, index, current_row_ids, "table_row")
        for row in rows:
            doc_id = row_document_id(TABLE_NAME, row["id"])
            try:
                payload = canonical_row_payload(row)
                content_hash = sha256_text(payload)
                chunks = [row_as_document(row)]
                status = process_document(
                    conn=conn,
                    index=index,
                    embeddings=embeddings,
                    document_id=doc_id,
                    content_hash=content_hash,
                    source_type="table_row",
                    chunks=chunks,
                )
                logger.info("row %s -> %s", doc_id, status)
            except Exception as exc:
                logger.exception("outer failure row %s: %s", doc_id, exc)
                upsert_document_state(
                    conn,
                    document_id=doc_id,
                    content_hash="",
                    version=0,
                    chunk_count=0,
                    status="failed",
                    embedding_model=EMBEDDING_MODEL,
                    source_type="table_row",
                    last_error=str(exc)[:1000],
                )

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Harold Pinecone incremental ingestion")
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--rows-only", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--policy-dir", default=None)
    args = parser.parse_args()
    run_ingestion(
        files_only=args.files_only,
        rows_only=args.rows_only,
        limit_rows=args.limit_rows,
        policy_dir=args.policy_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test files-only run**

```bash
cd /home/kai/repo/harold && PYTHONPATH=scripts python scripts/pinecone_ingestion.py --files-only
```

Expected: both policy files `success` or `skipped` on re-run; state rows in MySQL; vectors in Pinecone.

- [ ] **Step 4: Test limited rows**

```bash
PYTHONPATH=scripts python scripts/pinecone_ingestion.py --rows-only --limit-rows 5
```

Expected: 5 row documents processed; re-run skips unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/pinecone_ingestion.py
git commit -m "$(cat <<'EOF'
feat: wire upsert-then-delete ingestion main loop

EOF
)"
```

---

### Task 7: Failure drill + idempotency check

**Files:**
- None required (manual verification); optional note in README only if user asks (skip docs unless requested)

- [ ] **Step 1: Confirm skip on second files-only run**

Run twice; logs should show `skipped` for unchanged files.

- [ ] **Step 2: Deliberate failure**

Temporarily break upsert (e.g. set invalid batch) OR stop after forcing `raise` inside `process_document` after a successful prior ingest; confirm:
- MySQL `status=failed`
- Previous vector IDs still fetchable in Pinecone
- Fix and re-run → `success` with new version; old IDs gone

- [ ] **Step 3: Final unit test suite**

Run: `python -m pytest tests/test_ingestion_helpers.py -v`  
Expected: PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add scripts/pinecone_ingestion.py tests/test_ingestion_helpers.py
git commit -m "$(cat <<'EOF'
fix: harden ingestion failure path after failure drill

EOF
)"
```

---

## Plan self-review

1. **Spec coverage:** Single model, MySQL state, upsert-then-delete, hierarchical IDs, index create/validate-no-delete, file + row sources, metadata, per-file errors — all mapped to Tasks 1–7.
2. **Placeholders:** None left; concrete code in each task.
3. **Type consistency:** `Document`, `vector_id`, `process_document`, `STATE_TABLE`, `TABLE_NAME` names aligned across tasks.
4. **Scale note:** Full 32k-row ingest is intentional but slow on CPU; always validate with `--limit-rows` first.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-19-pinecone-ingestion.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
