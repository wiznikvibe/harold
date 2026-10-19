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
