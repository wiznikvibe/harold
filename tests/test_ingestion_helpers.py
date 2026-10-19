import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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


def test_row_to_text_skips_nulls():
    text = ing.row_to_text({"id": 1, "RISK_CLS_TXT": "HIGH", "x": None})
    assert "id: 1" in text
    assert "RISK_CLS_TXT: HIGH" in text
    assert "x:" not in text


def test_init_state_schema_sql_contains_table_name():
    statements = ing.state_schema_statements()
    assert any(ing.STATE_TABLE in s for s in statements)
    assert any("content_hash" in s for s in statements)
    assert any("chunk_count" in s for s in statements)


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


def test_failure_state_values_preserves_prior():
    prior = {
        "content_hash": "deadbeef",
        "version": 3,
        "chunk_count": 12,
        "embedding_model": "custom/model",
    }
    ch, ver, cc, em = ing.failure_state_values(prior)
    assert (ch, ver, cc, em) == ("deadbeef", 3, 12, "custom/model")


def test_failure_state_values_new_document():
    ch, ver, cc, em = ing.failure_state_values(None, new_content_hash="abc")
    assert ch == "abc"
    assert ver == 0
    assert cc == 0
    assert em == ing.EMBEDDING_MODEL


def test_failure_state_values_prior_ignores_new_content_hash():
    prior = {
        "content_hash": "keep-me",
        "version": 2,
        "chunk_count": 5,
        "embedding_model": "m",
    }
    ch, ver, cc, em = ing.failure_state_values(prior, new_content_hash="would-wipe")
    assert (ch, ver, cc, em) == ("keep-me", 2, 5, "m")


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self.last_execute = None

    def execute(self, sql, params=None):
        self.last_execute = (sql, params)
        self._conn.last_sql = sql
        self._conn.last_params = params

    def fetchone(self):
        if self._conn.fetchone_result is not None:
            return self._conn.fetchone_result
        return None

    def fetchall(self):
        return []


class FakeConn:
    def __init__(self, *, fetchone_result=None):
        self.fetchone_result = fetchone_result
        self.last_sql = None
        self.last_params = None

    def close(self):
        pass

    @contextmanager
    def cursor(self):
        yield FakeCursor(self)


def test_upsert_outer_failure_state_preserves_prior_version():
    prior = {
        "content_hash": "deadbeef",
        "version": 7,
        "chunk_count": 99,
        "embedding_model": "custom/model",
    }
    conn = FakeConn(fetchone_result=prior)
    ing.upsert_outer_failure_state(
        conn,
        "SOP.md",
        source_type="file",
        last_error="parse blew up",
        new_content_hash="new-hash-should-not-win",
    )
    assert conn.last_params is not None
    (
        document_id,
        content_hash,
        version,
        chunk_count,
        status,
        embedding_model,
        source_type,
        last_error,
    ) = conn.last_params
    assert document_id == "SOP.md"
    assert content_hash == "deadbeef"
    assert version == 7
    assert chunk_count == 99
    assert status == "failed"
    assert embedding_model == "custom/model"
    assert source_type == "file"
    assert last_error == "parse blew up"


def test_upsert_outer_failure_state_new_document_uses_new_hash():
    conn = FakeConn(fetchone_result=None)
    ing.upsert_outer_failure_state(
        conn,
        "new.md",
        source_type="file",
        last_error="read failed",
        new_content_hash="fresh",
    )
    _, content_hash, version, chunk_count, status, embedding_model, _, _ = conn.last_params
    assert content_hash == "fresh"
    assert version == 0
    assert chunk_count == 0
    assert status == "failed"
    assert embedding_model == ing.EMBEDDING_MODEL


def test_should_process_files_missing_policy_dir(tmp_path):
    missing = tmp_path / "nope"
    ok, reason = ing.should_process_files(missing, {"a.md"}, set())
    assert ok is False
    assert reason == "policy_dir_not_directory"


def test_should_process_files_empty_discovery_with_tracked(tmp_path):
    policy = tmp_path / "policies"
    policy.mkdir()
    ok, reason = ing.should_process_files(policy, set(), {"SOP.md"})
    assert ok is False
    assert reason == "empty_discovery_with_tracked"


def test_should_process_files_empty_discovery_no_tracked_ok(tmp_path):
    policy = tmp_path / "policies"
    policy.mkdir()
    ok, reason = ing.should_process_files(policy, set(), set())
    assert ok is True
    assert reason == ""


def test_should_process_files_happy_path(tmp_path):
    policy = tmp_path / "policies"
    policy.mkdir()
    ok, reason = ing.should_process_files(policy, {"SOP.md"}, {"old.md"})
    assert ok is True
    assert reason == ""


def test_run_ingestion_skips_reconcile_when_policy_dir_missing(tmp_path):
    with patch.object(ing, "load_config") as load_cfg, patch.object(
        ing, "get_mysql_connection"
    ) as get_conn, patch.object(ing, "init_state_schema"), patch.object(
        ing, "build_embeddings"
    ), patch.object(
        ing, "embedding_dimension", return_value=8
    ), patch.object(
        ing, "Pinecone"
    ), patch.object(
        ing, "ensure_pinecone_index"
    ), patch.object(
        ing, "list_document_ids", return_value=["SOP.md"]
    ), patch.object(
        ing, "reconcile_deletions"
    ) as reconcile, patch.object(
        ing, "discover_policy_files"
    ) as discover:
        load_cfg.return_value = {
            "policy_dir": str(tmp_path / "missing"),
            "embedding_model": ing.EMBEDDING_MODEL,
            "pinecone_api_key": "k",
            "index_name": "idx",
        }
        fake_conn = FakeConn()
        get_conn.return_value = fake_conn
        ing.run_ingestion(files_only=True, policy_dir=str(tmp_path / "missing"))
        reconcile.assert_not_called()
        discover.assert_not_called()
