# Pinecone Incremental Ingestion Pipeline — Design

**Date:** 2026-10-19  
**Status:** Approved (user assumed remaining section approval)  
**Repo:** harold

## Goal

Build a production-grade, incremental ingestion pipeline that loads policy Markdown (and other supported file types) from `datasource/policy/` plus telemetry rows from MySQL into a **single** Pinecone index, with no temporary data-loss windows, using **one** embedding model only: `google/embeddinggemma-300m`.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Script layout | Single runnable script: `scripts/pinecone_ingestion.py` |
| State store | MySQL (not Postgres) on `100.31.187.36:3306`, database `logistics` |
| Tabular source | Table `dynamic_supply_chain_logistics` only |
| Row granularity | One document per row: `dynamic_supply_chain_logistics-{id}` |
| File source | `datasource/policy/` (md/txt/pdf/csv/xlsx) |
| Pinecone index | `harold-logistics` (create if missing; never auto-delete) |
| Embedding model | `google/embeddinggemma-300m` only (Hugging Face) |
| Update order | Upsert new version → verify → delete old version → mark success |
| Vector IDs | `{document_id}-v{version}-c{chunk_index}` |

## Architecture

```
datasource/policy/*  ──┐
                       ├──► parse/chunk ──► embed (EmbeddingGemma) ──► Pinecone harold-logistics
MySQL table rows     ──┘         ▲
                                 │
                    MySQL ingestion_document_state
                    (hash, version, chunk_count, status)
```

- Credentials and connection settings come from `.env` (never hardcode secrets in the script).
- Same MySQL database hosts both the telemetry table and the ingestion state table.
- Index dimension is taken from the embedding model at runtime. If an existing index has a mismatched dimension, raise and stop.

## Components (one file)

1. **Config & setup** — load env; init HuggingFace embeddings; resolve model dimension; create/validate Pinecone index.
2. **MySQL state helpers** — create schema; get/upsert state; mark deleted; list known document IDs by `source_type`.
3. **Parsers** — files → LangChain-style `Document` list; table rows → one text document per row.
4. **Embed + upsert** — hierarchical IDs; batched upsert with retries; return upserted count.
5. **Main loop** — discover sources; delete removed docs; skip unchanged; process changed with upsert-then-delete; record failures without deleting old vectors.

## Data model

### State table: `ingestion_document_state`

| Column | Type | Notes |
|--------|------|-------|
| `document_id` | VARCHAR(512) PK | File name or `dynamic_supply_chain_logistics-{id}` |
| `content_hash` | CHAR(64) | SHA-256 hex of canonical content |
| `version` | INT | Starts at 1; increments on change |
| `chunk_count` | INT | Used to reconstruct vector IDs for delete |
| `status` | ENUM/VARCHAR | `success` / `failed` |
| `embedding_model` | VARCHAR(128) | Always `google/embeddinggemma-300m` on success |
| `source_type` | VARCHAR(32) | `file` / `table_row` |
| `last_error` | TEXT NULL | Short error on failure |
| `created_at` | TIMESTAMP | |
| `updated_at` | TIMESTAMP | |

Only set `status=success` after upsert count is verified and old vectors (if any) are deleted.

### Vector metadata (every vector)

`document_id`, `source_file` (filename or table name), `version`, `chunk_index`, `embedding_model`, `file_format`, `document_type`, `ingested_at`, plus when available: `page_number`, `row_index`, `row_id` (MySQL primary key).

### Content hashing

- **Files:** SHA-256 of raw file bytes.
- **Rows:** SHA-256 of a stable canonical serialization of column values (sorted keys or fixed column order; nulls explicit).

## Processing algorithm

1. Initialize MySQL schema + Pinecone index + embedding model.
2. Discover current file document IDs and current table-row document IDs.
3. For state rows whose `document_id` is absent from discovery → delete vectors for stored `version`/`chunk_count` → delete state row.
4. For each current document:
   - Compute `content_hash`.
   - If state exists with same hash and `status=success` → skip.
   - Else `new_version = previous.version + 1` (or 1 if new).
   - Parse → chunk → embed → upsert with IDs `{document_id}-v{new_version}-c{i}`.
   - If upserted count ≠ chunk count → record failure; leave old version.
   - Else delete previous version IDs (if any) → upsert state as success.
5. Per-document try/except: on error, store `failed` + truncated `last_error`; never delete old vectors.

## CLI / operational knobs

- Default: ingest all policy files + all table rows.
- `--limit-rows N` for safe testing against the ~32k-row table.
- Optional `--files-only` / `--rows-only` for partial runs (same safety rules).

## Error handling

- Transient Pinecone/network failures: retry with short backoff inside upsert batches.
- Permanent/file-level failures: log clearly; MySQL `failed`; previous Pinecone version remains queryable.
- Dimension mismatch on existing index: hard stop (no index deletion).

## Testing before full corpus

1. Run against 2 policy files only.
2. Run with `--limit-rows 10`.
3. Simulate failure (e.g. invalid API or forced exception after partial upsert) and confirm prior version IDs remain and state is `failed`.
4. Re-run; failed/changed docs retry; unchanged skip.

## Out of scope

- Dual embedding models or multiple indexes.
- Postgres state store (MySQL chosen deliberately).
- Auto-deletion of mismatched Pinecone indexes.
- Separate microservices / multi-file package layout.

## Env vars expected

- `PINECONE_API_KEY` (existing)
- `HUGGING_FACE_API_KEY` (existing)
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (add to `.env`; or a single `DATABASE_URL`)
- `PINECONE_INDEX_NAME` optional, default `harold-logistics`
- `EMBEDDING_MODEL` optional, default `google/embeddinggemma-300m`
