/**
 * Open Brain REST API - Kubernetes self-hosted Postgres version.
 *
 * This mirrors the Supabase Edge Function REST gateway enough for the
 * Next.js dashboard while using the same Postgres + pgvector database as
 * the self-hosted MCP server.
 */

import { Hono } from "hono";
import { z } from "zod";
import { Pool } from "postgres";

const DB_HOST = Deno.env.get("DB_HOST") || "127.0.0.1";
const DB_PORT = parseInt(Deno.env.get("DB_PORT") || "5432", 10);
const DB_NAME = Deno.env.get("DB_NAME") || "openbrain";
const DB_USER = Deno.env.get("DB_USER") || "postgres";
const DB_PASSWORD = Deno.env.get("DB_PASSWORD")!;

const EMBEDDING_API_BASE =
  Deno.env.get("EMBEDDING_API_BASE") || "https://openrouter.ai/api/v1";
const EMBEDDING_API_KEY =
  Deno.env.get("EMBEDDING_API_KEY") || Deno.env.get("OPENROUTER_API_KEY") || "";
const EMBEDDING_MODEL =
  Deno.env.get("EMBEDDING_MODEL") || "openai/text-embedding-3-small";

const CHAT_API_BASE = Deno.env.get("CHAT_API_BASE") || EMBEDDING_API_BASE;
const CHAT_API_KEY = Deno.env.get("CHAT_API_KEY") || EMBEDDING_API_KEY;
const CHAT_MODEL = Deno.env.get("CHAT_MODEL") || "openai/gpt-4o-mini";

const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY")!;
const PORT = parseInt(Deno.env.get("PORT") || "8000", 10);

const pool = new Pool(
  {
    hostname: DB_HOST,
    port: DB_PORT,
    database: DB_NAME,
    user: DB_USER,
    password: DB_PASSWORD,
  },
  20,
);

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-brain-key",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
};

type DbThought = {
  id: string;
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: Date | string;
  updated_at?: Date | string | null;
  type?: string | null;
  source_type?: string | null;
  importance?: number | string | null;
  quality_score?: number | string | null;
  sensitivity_tier?: string | null;
  status?: string | null;
  status_updated_at?: Date | string | null;
};

type NormalizedThought = {
  id: string;
  uuid: string;
  content: string;
  type: string;
  source_type: string;
  importance: number;
  quality_score: number;
  sensitivity_tier: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  status: string | null;
  status_updated_at: string | null;
};

const captureSchema = z.object({
  content: z.string().min(1),
  metadata: z.record(z.string(), z.unknown()).optional(),
  type: z.string().optional(),
  source_type: z.string().optional(),
  importance: z.number().min(0).max(100).optional(),
  quality_score: z.number().min(0).max(100).optional(),
  sensitivity_tier: z.string().optional(),
  status: z.string().nullable().optional(),
});

const updateSchema = z.object({
  content: z.string().min(1).optional(),
  type: z.string().optional(),
  importance: z.number().min(0).max(100).optional(),
  quality_score: z.number().min(0).max(100).optional(),
  sensitivity_tier: z.string().optional(),
  status: z.string().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

const searchSchema = z.object({
  query: z.string().min(1),
  mode: z.enum(["semantic", "text"]).default("semantic"),
  limit: z.number().int().min(1).max(100).default(25),
  page: z.number().int().min(1).default(1),
  threshold: z.number().min(0).max(1).default(0.35),
  exclude_restricted: z.boolean().default(true),
});

const reflectionSchema = z.object({
  trigger_context: z.string().optional().default(""),
  options: z.array(z.unknown()).optional().default([]),
  factors: z.array(z.unknown()).optional().default([]),
  conclusion: z.string().optional().default(""),
  confidence: z.number().min(0).max(1).optional().default(0.75),
  reflection_type: z.string().optional().default("reflection"),
  metadata: z.record(z.string(), z.unknown()).optional().default({}),
});

function auth(c: { req: { header: (name: string) => string | undefined; url: string } }) {
  const provided =
    c.req.header("x-brain-key") || new URL(c.req.url).searchParams.get("key");
  return Boolean(provided && provided === MCP_ACCESS_KEY);
}

function intParam(value: string | null, fallback: number, min = 0, max = 1000) {
  const parsed = Number.parseInt(value || "", 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function numberValue(value: unknown, fallback: number) {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function metadataOf(row: Pick<DbThought, "metadata">): Record<string, unknown> {
  return row.metadata && typeof row.metadata === "object" ? row.metadata : {};
}

function stringMeta(metadata: Record<string, unknown>, key: string): string | null {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function iso(value: Date | string | null | undefined, fallback?: Date | string) {
  const input = value ?? fallback ?? new Date();
  if (input instanceof Date) return input.toISOString();
  return new Date(input).toISOString();
}

function normalizeThought(
  row: DbThought,
  extra: Record<string, unknown> = {},
): NormalizedThought & Record<string, unknown> {
  const metadata = metadataOf(row);
  const type = row.type || stringMeta(metadata, "type") || "observation";
  const sourceType =
    row.source_type ||
    stringMeta(metadata, "source") ||
    stringMeta(metadata, "source_type") ||
    "unknown";
  const sensitivity =
    row.sensitivity_tier || stringMeta(metadata, "sensitivity_tier") || "standard";
  const createdAt = iso(row.created_at);
  const updatedAt = iso(row.updated_at, row.created_at);

  return {
    id: String(row.id),
    uuid: String(row.id),
    content: row.content,
    type,
    source_type: sourceType,
    importance: numberValue(row.importance, numberValue(metadata.importance, 50)),
    quality_score: numberValue(
      row.quality_score,
      numberValue(metadata.quality_score, 50),
    ),
    sensitivity_tier: sensitivity,
    metadata: {
      ...metadata,
      type,
      source: sourceType,
      source_type: sourceType,
    },
    created_at: createdAt,
    updated_at: updatedAt,
    status: row.status ?? null,
    status_updated_at: row.status_updated_at ? iso(row.status_updated_at) : null,
    ...extra,
  };
}

function isRestricted(row: DbThought | ReturnType<typeof normalizeThought>) {
  const metadata = metadataOf(row);
  return (
    row.sensitivity_tier === "restricted" ||
    stringMeta(metadata, "sensitivity_tier") === "restricted"
  );
}

function compactFingerprint(text: string) {
  return text.toLowerCase().replace(/\s+/g, " ").replace(/[^\w\s]/g, "").trim();
}

function tokenSimilarity(a: string, b: string) {
  const aTokens = new Set(a.toLowerCase().match(/[a-z0-9]{3,}/g) || []);
  const bTokens = new Set(b.toLowerCase().match(/[a-z0-9]{3,}/g) || []);
  if (aTokens.size === 0 || bTokens.size === 0) return 0;
  let intersection = 0;
  for (const token of aTokens) if (bTokens.has(token)) intersection += 1;
  const union = new Set([...aTokens, ...bTokens]).size;
  return intersection / union;
}

async function contentFingerprint(text: string) {
  const data = new TextEncoder().encode(compactFingerprint(text));
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function getEmbedding(text: string): Promise<number[]> {
  if (!EMBEDDING_API_KEY) throw new Error("EMBEDDING_API_KEY is not configured");
  const response = await fetch(`${EMBEDDING_API_BASE}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${EMBEDDING_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: text,
    }),
  });
  if (!response.ok) {
    throw new Error(`Embedding API failed: ${response.status} ${await response.text()}`);
  }
  const data = await response.json();
  return data.data[0].embedding;
}

async function extractMetadata(text: string): Promise<Record<string, unknown>> {
  if (!CHAT_API_KEY) return fallbackMetadata(text);
  const response = await fetch(`${CHAT_API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${CHAT_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            'Extract metadata from this Open Brain thought. Return JSON with "people", "topics", "action_items", "dates_mentioned", and "type". Type must be one of observation, task, idea, reference, person_note, decision, lesson, meeting, journal. Only extract what is explicit.',
        },
        { role: "user", content: text },
      ],
    }),
  });
  if (!response.ok) return fallbackMetadata(text);
  const data = await response.json();
  try {
    return JSON.parse(data.choices[0].message.content);
  } catch {
    return fallbackMetadata(text);
  }
}

function fallbackMetadata(text: string): Record<string, unknown> {
  const lower = text.toLowerCase();
  const type = /\b(todo|next step|ship|implement|fix|review|publish)\b/.test(lower)
    ? "task"
    : /\b(decided|decision|must|should)\b/.test(lower)
      ? "decision"
      : /\b(recipe|docs|reference|guide|url|http)\b/.test(lower)
        ? "reference"
        : "observation";
  const topics = [
    lower.includes("openclaw") ? "OpenClaw" : null,
    lower.includes("agent memory") ? "agent memory" : null,
    lower.includes("dashboard") ? "dashboard" : null,
    lower.includes("nate") ? "Nate Jones" : null,
  ].filter(Boolean);
  return {
    type,
    topics: topics.length ? topics : ["open brain"],
    people: [],
    action_items: [],
    dates_mentioned: [],
  };
}

function rowSelect(extra = "") {
  return `SELECT id::text, content, metadata, created_at, updated_at, type,
                 source_type, importance, quality_score, sensitivity_tier,
                 status, status_updated_at ${extra}
          FROM thoughts`;
}

function filterSql(url: URL, startIndex = 1) {
  const conditions: string[] = [];
  const params: unknown[] = [];
  let index = startIndex;

  if (url.searchParams.get("exclude_restricted") !== "false") {
    conditions.push(
      "COALESCE(sensitivity_tier, metadata->>'sensitivity_tier', 'standard') <> 'restricted'",
    );
  }
  const type = url.searchParams.get("type");
  if (type) {
    conditions.push(`COALESCE(type, metadata->>'type') = $${index++}`);
    params.push(type);
  }
  const sourceType = url.searchParams.get("source_type");
  if (sourceType) {
    conditions.push(
      `COALESCE(source_type, metadata->>'source_type', metadata->>'source') = $${index++}`,
    );
    params.push(sourceType);
  }
  const status = url.searchParams.get("status");
  if (status) {
    const statuses = status.split(",").map((s) => s.trim()).filter(Boolean);
    if (statuses.length === 1) {
      conditions.push(`status = $${index++}`);
      params.push(statuses[0]);
    } else if (statuses.length > 1) {
      conditions.push(`status = ANY($${index++})`);
      params.push(statuses);
    }
  }
  const importanceMin = url.searchParams.get("importance_min");
  if (importanceMin !== null) {
    conditions.push(`COALESCE(importance, 50) >= $${index++}`);
    params.push(Number(importanceMin));
  }
  const qualityMax = url.searchParams.get("quality_score_max");
  if (qualityMax !== null) {
    conditions.push(`COALESCE(quality_score, 50) <= $${index++}`);
    params.push(Number(qualityMax));
  }

  return {
    where: conditions.length ? `WHERE ${conditions.join(" AND ")}` : "",
    params,
    nextIndex: index,
  };
}

function sortSql(url: URL) {
  const requested = url.searchParams.get("sort") || "created_at";
  const columns: Record<string, string> = {
    created_at: "created_at",
    updated_at: "updated_at",
    importance: "COALESCE(importance, 50)",
    quality_score: "COALESCE(quality_score, 50)",
    type: "COALESCE(type, metadata->>'type')",
    source_type: "COALESCE(source_type, metadata->>'source_type', metadata->>'source')",
    status: "status",
  };
  const column = columns[requested] || columns.created_at;
  const direction = url.searchParams.get("order") === "asc" ? "ASC" : "DESC";
  return `ORDER BY ${column} ${direction} NULLS LAST`;
}

async function listThoughts(url: URL) {
  const page = intParam(url.searchParams.get("page"), 1, 1, 100000);
  const perPage = intParam(url.searchParams.get("per_page"), 25, 1, 100);
  const offset = (page - 1) * perPage;
  const filter = filterSql(url);
  const countClient = await pool.connect();
  try {
    const countResult = await countClient.queryObject<{ count: number }>(
      `SELECT COUNT(*)::int AS count FROM thoughts ${filter.where}`,
      filter.params,
    );
    const rowsResult = await countClient.queryObject<DbThought>(
      `${rowSelect()} ${filter.where} ${sortSql(url)}
       LIMIT $${filter.nextIndex} OFFSET $${filter.nextIndex + 1}`,
      [...filter.params, perPage, offset],
    );
    return {
      data: rowsResult.rows.map((row) => normalizeThought(row)),
      total: countResult.rows[0]?.count || 0,
      page,
      per_page: perPage,
    };
  } finally {
    countClient.release();
  }
}

async function getThought(id: string) {
  const client = await pool.connect();
  try {
    const result = await client.queryObject<DbThought>(
      `${rowSelect()} WHERE id = $1 LIMIT 1`,
      [id],
    );
    return result.rows[0] || null;
  } finally {
    client.release();
  }
}

async function stats(url: URL) {
  const allUrl = new URL(url);
  allUrl.searchParams.delete("page");
  allUrl.searchParams.set("per_page", "100");
  const thoughts = await listThoughts(allUrl);
  const types: Record<string, number> = {};
  const topics: Record<string, number> = {};
  for (const thought of thoughts.data) {
    types[thought.type] = (types[thought.type] || 0) + 1;
    const thoughtTopics = Array.isArray(thought.metadata["topics"])
      ? (thought.metadata["topics"] as string[])
      : [];
    for (const topic of thoughtTopics) topics[topic] = (topics[topic] || 0) + 1;
  }
  return {
    total_thoughts: thoughts.total,
    window_days: intParam(url.searchParams.get("days"), 0, 0, 3650) || "all",
    types,
    top_topics: Object.entries(topics)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([topic, count]) => ({ topic, count })),
  };
}

async function semanticSearch(body: z.infer<typeof searchSchema>) {
  const embedding = await getEmbedding(body.query);
  const embStr = `[${embedding.join(",")}]`;
  const matchCount = Math.min(100, Math.max(body.limit * body.page * 3, body.limit));
  const client = await pool.connect();
  try {
    const result = await client.queryObject<DbThought & { similarity: number }>(
      `${rowSelect(", 1 - (embedding <=> $1::vector) AS similarity")}
       WHERE embedding IS NOT NULL
         AND 1 - (embedding <=> $1::vector) >= $2
       ORDER BY embedding <=> $1::vector
       LIMIT $3`,
      [embStr, body.threshold, matchCount],
    );
    const ordered = result.rows
      .filter((row) => !body.exclude_restricted || !isRestricted(row))
      .map((row, index) =>
        normalizeThought(row, { similarity: row.similarity, rank: index + 1 })
      );
    const offset = (body.page - 1) * body.limit;
    return {
      results: ordered.slice(offset, offset + body.limit),
      count: Math.min(body.limit, Math.max(0, ordered.length - offset)),
      total: ordered.length,
      page: body.page,
      per_page: body.limit,
      total_pages: Math.max(1, Math.ceil(ordered.length / body.limit)),
      mode: "semantic",
    };
  } finally {
    client.release();
  }
}

async function textSearch(body: z.infer<typeof searchSchema>) {
  const offset = (body.page - 1) * body.limit;
  const params: unknown[] = [`%${body.query}%`];
  const restricted = body.exclude_restricted
    ? "AND COALESCE(sensitivity_tier, metadata->>'sensitivity_tier', 'standard') <> 'restricted'"
    : "";
  const client = await pool.connect();
  try {
    const countResult = await client.queryObject<{ count: number }>(
      `SELECT COUNT(*)::int AS count FROM thoughts
       WHERE content ILIKE $1 ${restricted}`,
      params,
    );
    const result = await client.queryObject<DbThought>(
      `${rowSelect()}
       WHERE content ILIKE $1 ${restricted}
       ORDER BY created_at DESC
       LIMIT $2 OFFSET $3`,
      [params[0], body.limit, offset],
    );
    return {
      results: result.rows.map((row, index) =>
        normalizeThought(row, { rank: offset + index + 1 })
      ),
      count: result.rows.length,
      total: countResult.rows[0]?.count || 0,
      page: body.page,
      per_page: body.limit,
      total_pages: Math.max(1, Math.ceil((countResult.rows[0]?.count || 0) / body.limit)),
      mode: "text",
    };
  } finally {
    client.release();
  }
}

async function createThought(body: z.infer<typeof captureSchema>) {
  const content = body.content.trim();
  const extracted = body.metadata ? body.metadata : await extractMetadata(content);
  const type = body.type || stringMeta(extracted, "type") || "observation";
  const sourceType = body.source_type || stringMeta(extracted, "source") || "dashboard";
  const sensitivity =
    body.sensitivity_tier || stringMeta(extracted, "sensitivity_tier") || "standard";
  const status =
    body.status !== undefined ? body.status : ["task", "idea"].includes(type) ? "new" : null;
  const metadata = {
    ...extracted,
    type,
    source: sourceType,
    source_type: sourceType,
    sensitivity_tier: sensitivity,
  };
  const embedding = await getEmbedding(content);
  const embStr = `[${embedding.join(",")}]`;
  const client = await pool.connect();
  try {
    const result = await client.queryObject<{ id: string }>(
      `INSERT INTO thoughts (
         content, embedding, metadata, type, source_type, importance,
         quality_score, sensitivity_tier, status, status_updated_at, updated_at
       )
       VALUES (
         $1, $2::vector, $3::jsonb, $4, $5, $6, $7, $8, $9,
         CASE WHEN $9::text IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END,
         CURRENT_TIMESTAMP
       )
       RETURNING id::text`,
      [
        content,
        embStr,
        JSON.stringify(metadata),
        type,
        sourceType,
        body.importance ?? numberValue(extracted.importance, 50),
        body.quality_score ?? numberValue(extracted.quality_score, 70),
        sensitivity,
        status,
      ],
    );
    return {
      thought_id: result.rows[0].id,
      action: "created",
      type,
      sensitivity_tier: sensitivity,
      content_fingerprint: await contentFingerprint(content),
      message: "Thought captured",
    };
  } finally {
    client.release();
  }
}

const app = new Hono();

app.options("*", (c) => c.text("ok", 200, corsHeaders));

app.use("*", async (c, next) => {
  if (!auth(c)) return c.json({ error: "Invalid or missing access key" }, 401, corsHeaders);
  await next();
});

app.get("/health", (c) =>
  c.json({ ok: true, status: "ok", service: "open-brain-rest-postgres" }, 200, corsHeaders)
);

app.get("/count", async (c) => {
  const data = await listThoughts(new URL(c.req.url));
  return c.json({ count: data.total, total: data.total }, 200, corsHeaders);
});

app.get("/stats", async (c) => {
  try {
    return c.json(await stats(new URL(c.req.url)), 200, corsHeaders);
  } catch (error) {
    return c.json({ error: error instanceof Error ? error.message : "Stats failed" }, 500, corsHeaders);
  }
});

app.get("/thoughts", async (c) => {
  try {
    return c.json(await listThoughts(new URL(c.req.url)), 200, corsHeaders);
  } catch (error) {
    return c.json({ error: error instanceof Error ? error.message : "List failed" }, 500, corsHeaders);
  }
});

app.get("/thought/:id", async (c) => {
  const row = await getThought(c.req.param("id"));
  if (!row) return c.json({ error: "Thought not found" }, 404, corsHeaders);
  const excludeRestricted = new URL(c.req.url).searchParams.get("exclude_restricted") !== "false";
  if (excludeRestricted && isRestricted(row)) {
    return c.json({ error: "Restricted thought" }, 403, corsHeaders);
  }
  return c.json(normalizeThought(row), 200, corsHeaders);
});

app.put("/thought/:id", async (c) => {
  try {
    const parsed = updateSchema.safeParse(await c.req.json());
    if (!parsed.success) {
      return c.json({ error: "Invalid update payload", details: parsed.error.flatten() }, 400, corsHeaders);
    }
    const existing = await getThought(c.req.param("id"));
    if (!existing) return c.json({ error: "Thought not found" }, 404, corsHeaders);
    const metadata = { ...metadataOf(existing), ...(parsed.data.metadata || {}) };
    if (parsed.data.type) metadata.type = parsed.data.type;

    const sets = ["metadata = $1::jsonb", "updated_at = CURRENT_TIMESTAMP"];
    const params: unknown[] = [JSON.stringify(metadata)];
    let index = 2;
    if (parsed.data.content !== undefined) {
      const embedding = await getEmbedding(parsed.data.content);
      sets.push(`content = $${index++}`);
      params.push(parsed.data.content);
      sets.push(`embedding = $${index++}::vector`);
      params.push(`[${embedding.join(",")}]`);
    }
    for (const [column, value] of Object.entries({
      type: parsed.data.type,
      importance: parsed.data.importance,
      quality_score: parsed.data.quality_score,
      sensitivity_tier: parsed.data.sensitivity_tier,
      status: parsed.data.status,
    })) {
      if (value !== undefined) {
        sets.push(`${column} = $${index++}`);
        params.push(value);
      }
    }
    if (parsed.data.status !== undefined) {
      sets.push("status_updated_at = CURRENT_TIMESTAMP");
    }
    params.push(c.req.param("id"));
    const client = await pool.connect();
    try {
      await client.queryObject(`UPDATE thoughts SET ${sets.join(", ")} WHERE id = $${index}`, params);
    } finally {
      client.release();
    }
    return c.json({ id: c.req.param("id"), action: "updated", message: "Thought updated" }, 200, corsHeaders);
  } catch (error) {
    return c.json({ error: error instanceof Error ? error.message : "Update failed" }, 500, corsHeaders);
  }
});

app.delete("/thought/:id", async (c) => {
  const client = await pool.connect();
  try {
    await client.queryObject("DELETE FROM thoughts WHERE id = $1", [c.req.param("id")]);
    return c.json({ id: c.req.param("id"), action: "deleted", message: "Thought deleted" }, 200, corsHeaders);
  } finally {
    client.release();
  }
});

app.post("/capture", async (c) => {
  try {
    const parsed = captureSchema.safeParse(await c.req.json());
    if (!parsed.success) {
      return c.json({ error: "Invalid capture payload", details: parsed.error.flatten() }, 400, corsHeaders);
    }
    return c.json(await createThought(parsed.data), 200, corsHeaders);
  } catch (error) {
    return c.json({ error: error instanceof Error ? error.message : "Capture failed" }, 500, corsHeaders);
  }
});

app.post("/search", async (c) => {
  try {
    const parsed = searchSchema.safeParse(await c.req.json());
    if (!parsed.success) {
      return c.json({ error: "Invalid search payload", details: parsed.error.flatten() }, 400, corsHeaders);
    }
    return c.json(
      parsed.data.mode === "text" ? await textSearch(parsed.data) : await semanticSearch(parsed.data),
      200,
      corsHeaders,
    );
  } catch (error) {
    return c.json({ error: error instanceof Error ? error.message : "Search failed" }, 500, corsHeaders);
  }
});

app.get("/duplicates", async (c) => {
  const url = new URL(c.req.url);
  const threshold = Number(url.searchParams.get("threshold") || 0.85);
  const limit = intParam(url.searchParams.get("limit"), 50, 1, 100);
  const offset = intParam(url.searchParams.get("offset"), 0, 0, 10000);
  const listUrl = new URL(url);
  listUrl.searchParams.set("per_page", "250");
  listUrl.searchParams.set("page", "1");
  listUrl.searchParams.delete("quality_score_max");
  const thoughts = (await listThoughts(listUrl)).data;
  const pairs = [];
  for (let i = 0; i < thoughts.length; i += 1) {
    for (let j = i + 1; j < thoughts.length; j += 1) {
      const a = thoughts[i];
      const b = thoughts[j];
      const exact = compactFingerprint(a.content) === compactFingerprint(b.content);
      const similarity = exact ? 1 : tokenSimilarity(a.content, b.content);
      if (similarity >= threshold) {
        pairs.push({
          thought_id_a: a.id,
          thought_id_b: b.id,
          similarity,
          content_a: a.content,
          content_b: b.content,
          type_a: a.type,
          type_b: b.type,
          quality_a: a.quality_score,
          quality_b: b.quality_score,
          created_a: a.created_at,
          created_b: b.created_at,
        });
      }
    }
  }
  pairs.sort((a, b) => b.similarity - a.similarity);
  return c.json({ pairs: pairs.slice(offset, offset + limit), threshold, limit, offset }, 200, corsHeaders);
});

app.get("/thought/:id/connections", async (c) => {
  const current = await getThought(c.req.param("id"));
  if (!current) return c.json({ connections: [] }, 200, corsHeaders);
  const currentMeta = metadataOf(current);
  const currentTopics = new Set(Array.isArray(currentMeta.topics) ? currentMeta.topics : []);
  const currentPeople = new Set(Array.isArray(currentMeta.people) ? currentMeta.people : []);
  if (!currentTopics.size && !currentPeople.size) {
    return c.json({ connections: [] }, 200, corsHeaders);
  }
  const url = new URL(c.req.url);
  const limit = intParam(url.searchParams.get("limit"), 20, 1, 50);
  const listUrl = new URL(url);
  listUrl.searchParams.set("per_page", "100");
  const thoughts = (await listThoughts(listUrl)).data.filter((thought) => thought.id !== current.id);
  const connections = thoughts
    .map((thought) => {
      const topics = Array.isArray(thought.metadata["topics"])
        ? (thought.metadata["topics"] as unknown[])
        : [];
      const people = Array.isArray(thought.metadata["people"])
        ? (thought.metadata["people"] as unknown[])
        : [];
      const overlap =
        topics.filter((topic) => currentTopics.has(topic)).length +
        people.filter((person) => currentPeople.has(person)).length;
      return { ...thought, overlap };
    })
    .filter((thought) => thought.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap)
    .slice(0, limit);
  return c.json({ connections }, 200, corsHeaders);
});

app.get("/thought/:id/reflection", (_c) => _c.json({ reflections: [] }, 200, corsHeaders));
app.post("/thought/:id/reflection", async (c) => {
  const parsed = reflectionSchema.safeParse(await c.req.json());
  if (!parsed.success) {
    return c.json({ error: "Invalid reflection payload", details: parsed.error.flatten() }, 400, corsHeaders);
  }
  return c.json({ ...parsed.data, id: 0, thought_id: c.req.param("id"), created_at: new Date().toISOString() }, 200, corsHeaders);
});
app.get("/ingestion-jobs", (c) => c.json({ jobs: [], count: 0 }, 200, corsHeaders));
app.get("/ingestion-jobs/:id", (c) => c.json({ job: null, items: [] }, 200, corsHeaders));
app.post("/ingestion-jobs/:id/execute", (c) => c.json({ job_id: c.req.param("id"), status: "not_configured" }, 200, corsHeaders));
app.post("/ingest", async (c) => {
  const body = await c.req.json().catch(() => ({}));
  const text = String(body.text || "").trim();
  if (!text) return c.json({ error: "text is required" }, 400, corsHeaders);
  const result = await createThought({ content: text, source_type: "dashboard_ingest" });
  return c.json({ job_id: 0, status: "complete", extracted_count: 1, thought_id: result.thought_id }, 200, corsHeaders);
});

Deno.serve({ port: PORT }, app.fetch);
