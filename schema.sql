-- Run this once in your Supabase SQL editor to create the tables.

-- Enable pgvector extension (already available on Supabase)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    file_name    TEXT        NOT NULL,
    file_type    TEXT        NOT NULL,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status       TEXT        NOT NULL DEFAULT 'uploaded',
    storage_path TEXT        NOT NULL,
    total_units  INTEGER
);

CREATE TABLE IF NOT EXISTS units (
    unit_id     TEXT PRIMARY KEY,
    document_id TEXT        NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    unit_type   TEXT,
    unit_number INTEGER,
    title       TEXT,
    raw_text    TEXT,
    clean_text  TEXT,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT        NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    unit_id     TEXT        NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    chunk_index INTEGER,
    text        TEXT,
    lang        TEXT,
    unit_number INTEGER,
    unit_type   TEXT,
    metadata    JSONB,
    embedding   vector(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_units_document_id  ON units(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id    ON chunks(chunk_id);
-- Cosine similarity index for fast vector search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding   ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
