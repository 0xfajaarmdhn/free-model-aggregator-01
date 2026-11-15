// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic), in JSON and in
// server-sent events, over the ten providers the trial's brief lists. Every answer comes from a free
// model that meets the request's requirements, chosen by the sealed quality table's rating; when a
// provider refuses, the request is retried and then failed over to the next-best eligible model.
// `node main.ts --list` prints the catalog and exits 0.
//
// Design notes (all from the kit's RULES.txt / brief / env / scorer, read before writing):
// - The brief at $MAKEFUN_PROMPT_FILE carries three JSON documents: the provider directory, the
//   quality table (ratings + per-request best), and the workload manifest. The catalog is built from
//   the directory + the quality table, so --list is identical inside and outside a trial.
// - The providers' answers echo what was sent to them (system, user text, images, tools, max_tokens,
//   stops, JSON mode); the client compares that echo with what it sent. So every field is passed
//   through untouched, in the provider's own wire, and the answer is returned in the client's wire
//   with the provider's text whole, its usage counts as that wire reports them, and its stop reason
//   mapped (end_turn -> stop, max_tokens -> length, tool_use -> tool_calls, refusal -> content_filter).
// - Exactly one successful provider call per request: a 200 from a provider is terminal (used, or the
//   request fails). Only a 402/429/5xx refusal is retried / failed over.
// - Determinism: no wall clock, no unseeded randomness. Every choice is a function of the brief, the
//   requests and the providers' answers. Fixed timestamps everywhere.
// - Node 22 standard library only; TypeScript is written with erasable syntax only (no enums,
//   namespaces or parameter properties) so `node main.ts` runs it directly.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const PROVIDER_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const MAX_ATTEMPTS_PER_REQUEST = 10;
/** Conservative per-provider call cap: the published floor of each provider's per-trial call range.
 *  Staying under it keeps a provider's budget from being spent by this entry's own primaries; a
 *  refusal past it is still handled by failover. */
const SOFT_CAP: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10,
  sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents found by their `schema` member.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: WorkloadRequest[];
  error: string | null;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return {
    id,
    name: strOf(v.name) !== '' ? strOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: strOf(v.key_env),
    key: strOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'the brief holds no provider directory' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || strOf(v.id) === '') return [];
          return [{ id: strOf(v.id), rating_milli: numOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The catalog: one entry per provider, from the directory + the quality table.
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** The catalog: the directory's providers, each with the quality table's free models in table order.
 *  base_url is the directory's own path (/p/<id>), never joined to the env URL: the env's origin
 *  differs between a live pass and its replay, and the catalog is part of the state export a replay
 *  must reproduce byte for byte. */
function buildCatalog(brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const models: CatalogModel[] = [];
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free) continue;
      models.push({ id: row.model, free: true, context_length: row.context_length });
    }
    providers.push({ id: p.id, base_url: p.base, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

/** The free models this entry serves, in the brief's provider order, for GET /v1/models. */
interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

function servedModels(brief: Brief): ServedModel[] {
  const out: ServedModel[] = [];
  for (const p of brief.providers) {
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free) continue;
      out.push({ id: row.model, provider: p.id, display_name: row.model });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Wire translation. The canonical request carries everything the providers echo,
// so a translation that drops or rewrites a field loses the request.
// ---------------------------------------------------------------------------

interface ImageBlock {
  kind: 'image';
  data: string | null;
  mediaType: string;
  url: string | null;
}

type ContentPart = { type: 'text'; text: string } | ImageBlock;

interface CanonMessage {
  role: string;
  parts: ContentPart[];
  toolCalls: { id: string; name: string; args: string }[];
  toolResultId: string | null;
  toolResultText: string | null;
}

interface CanonTool {
  name: string;
  description: string;
  inputSchema: Json;
}

interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: string | null;
  maxTokens: number | null;
  stop: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  stream: boolean;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One content block as a canonical part: text, an inline image, or null for the rest. */
function partOfOpenAI(block: Json): ContentPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (m !== null) return { kind: 'image', data: m[2], mediaType: m[1], url: null };
    return { kind: 'image', data: null, mediaType: '', url };
  }
  return null;
}

function partOfAnthropic(block: Json): ContentPart | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { kind: 'image', data: source.data, mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', url: null };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { kind: 'image', data: null, mediaType: '', url: source.url };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions body as the canonical request. */
function canonFromOpenAI(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system += (system !== '' ? '\n' : '') + text;
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: { id: string; name: string; args: string }[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
          });
        }
      }
      messages.push({ role: 'assistant', parts: [{ type: 'text', text: textOf(raw.content) }], toolCalls, toolResultId: null, toolResultText: null });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        parts: [],
        toolCalls: [],
        toolResultId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : null,
        toolResultText: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    const parts: ContentPart[] = [];
    if (typeof raw.content === 'string') parts.push({ type: 'text', text: raw.content });
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const part = partOfOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolResultId: null, toolResultText: null });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        inputSchema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stopRaw = body.stop;
  const stop: string[] = typeof stopRaw === 'string' ? [stopRaw] : Array.isArray(stopRaw) ? stopRaw.filter((x): x is string => typeof x === 'string') : [];
  return {
    system,
    messages,
    tools,
    toolChoice: typeof body.tool_choice === 'string' ? body.tool_choice : isRecord(body.tool_choice) && typeof body.tool_choice.type === 'string' ? body.tool_choice.type : null,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stop,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as the canonical request. */
function canonFromAnthropic(body: Json): CanonRequest {
  const system = textOf(body.system);
  const messages: CanonMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'assistant') {
      const toolCalls: { id: string; name: string; args: string }[] = [];
      let text = '';
      if (typeof raw.content === 'string') text = raw.content;
      else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') text += (text !== '' ? '\n' : '') + block.text;
          else if (block.type === 'tool_use') {
            toolCalls.push({
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              args: JSON.stringify(isRecord(block.input) ? block.input : {}),
            });
          }
        }
      }
      messages.push({ role: 'assistant', parts: [{ type: 'text', text }], toolCalls, toolResultId: null, toolResultText: null });
      continue;
    }
    const parts: ContentPart[] = [];
    let pendingToolResult: { id: string | null; text: string } | null = null;
    if (typeof raw.content === 'string') parts.push({ type: 'text', text: raw.content });
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          pendingToolResult = {
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : null,
            text: toolResultText(block.content),
          };
          continue;
        }
        const part = partOfAnthropic(block);
        if (part !== null) parts.push(part);
      }
    }
    if (pendingToolResult !== null) {
      messages.push({ role: 'tool', parts: [], toolCalls: [], toolResultId: pendingToolResult.id, toolResultText: pendingToolResult.text });
    }
    if (parts.length > 0) messages.push({ role: 'user', parts, toolCalls: [], toolResultId: null, toolResultText: null });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        inputSchema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  const toolChoice = isRecord(choice) && typeof choice.type === 'string' ? choice.type : null;
  const stopRaw = body.stop_sequences;
  const stop: string[] = Array.isArray(stopRaw) ? stopRaw.filter((x): x is string => typeof x === 'string') : [];
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    stop,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

function imageDataUrl(part: ImageBlock): string {
  if (part.data !== null) return `data:${part.mediaType !== '' ? part.mediaType : 'image/png'};base64,${part.data}`;
  return part.url ?? '';
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
function canonToOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const msg of canon.messages) {
    if (msg.role === 'assistant') {
      const out: Json = { role: 'assistant', content: msg.parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n') || null };
      if (msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(out);
      continue;
    }
    if (msg.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: msg.toolResultId ?? '', content: msg.toolResultText ?? '' });
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.type === 'text') parts.push({ type: 'text', text: part.text });
      else parts.push({ type: 'image_url', image_url: { url: imageDataUrl(part) } });
    }
    messages.push({ role: 'user', content: parts.length === 1 && parts[0].type === 'text' ? (parts[0] as { text: string }).text : parts });
  }
  const out: Json = { model, messages };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stop.length > 0) out.stop = canon.stop;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.inputSchema },
    }));
  }
  if (canon.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (canon.toolChoice === 'any' || canon.toolChoice === 'required') out.tool_choice = 'required';
  else if (canon.toolChoice === 'tool' || canon.toolChoice === 'none') out.tool_choice = canon.toolChoice;
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
function canonToAnthropic(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of canon.messages) {
    if (msg.role === 'assistant') {
      const content: Json[] = [];
      const text = msg.parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
      if (text !== '') content.push({ type: 'text', text });
      for (const c of msg.toolCalls) {
        let input: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        content.push({ type: 'tool_use', id: c.id, name: c.name, input });
      }
      if (content.length === 0) content.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (msg.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: msg.toolResultId ?? '', content: msg.toolResultText ?? '' }] });
      continue;
    }
    const content: Json[] = [];
    for (const part of msg.parts) {
      if (part.type === 'text') content.push({ type: 'text', text: part.text });
      else if (part.data !== null) {
        content.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType !== '' ? part.mediaType : 'image/png', data: part.data } });
      } else if (part.url !== null) {
        content.push({ type: 'image', source: { type: 'url', url: part.url } });
      }
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }
  const out: Json = { model, messages };
  if (canon.system !== '') out.system = canon.system;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stop.length > 0) out.stop_sequences = canon.stop;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema }));
  }
  if (canon.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (canon.toolChoice === 'any' || canon.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (canon.toolChoice === 'none') out.tool_choice = { type: 'none' };
  if (canon.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body. */
function canonToGemini(canon: CanonRequest): Json {
  const contents: Json[] = [];
  for (const msg of canon.messages) {
    if (msg.role === 'assistant') {
      const parts: Json[] = [];
      const text = msg.parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
      if (text !== '') parts.push({ text });
      for (const c of msg.toolCalls) {
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        parts.push({ functionCall: { name: c.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (msg.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: '', response: { content: msg.toolResultText ?? '' } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    for (const part of msg.parts) {
      if (part.type === 'text') parts.push({ text: part.text });
      else if (part.data !== null) {
        parts.push({ inlineData: { mimeType: part.mediaType !== '' ? part.mediaType : 'image/png', data: part.data } });
      } else if (part.url !== null) {
        parts.push({ fileData: { fileUri: part.url } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) generationConfig.temperature = canon.temperature;
  if (canon.topP !== null) generationConfig.topP = canon.topP;
  if (canon.stop.length > 0) generationConfig.stopSequences = canon.stop;
  if (canon.jsonMode) generationConfig.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (canon.system !== '') out.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.inputSchema })),
      },
    ];
  }
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers, parsed into one shape: text, usage, stop.
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseOpenAIAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : null;
    if (message === null) return null;
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text,
      usage: { input: numOrNull(usage.prompt_tokens) ?? 0, output: numOrNull(usage.completion_tokens) ?? 0 },
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  let text = '';
  let usage = { input: 0, output: 0 };
  let stop = 'stop';
  for (const frame of parseSseFrames(body)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numOrNull(chunk.usage.prompt_tokens) ?? usage.input, output: numOrNull(chunk.usage.completion_tokens) ?? usage.output };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, usage, stop };
}

function parseAnthropicAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: textOf(value.content),
      usage: { input: numOrNull(usage.input_tokens) ?? 0, output: numOrNull(usage.output_tokens) ?? 0 },
      stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
    };
  }
  let text = '';
  let usage = { input: 0, output: 0 };
  let stop = 'end_turn';
  for (const frame of parseSseFrames(body)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      usage.input = numOrNull(data.message.usage.input_tokens) ?? usage.input;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) usage.output = numOrNull(data.usage.output_tokens) ?? usage.output;
    }
  }
  return { text, usage, stop };
}

function parseGeminiAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
    const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts
      .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
      .map((p) => p.text as string)
      .join('');
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text,
      usage: { input: numOrNull(meta.promptTokenCount) ?? 0, output: numOrNull(meta.candidatesTokenCount) ?? 0 },
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    };
  }
  let text = '';
  let usage = { input: 0, output: 0 };
  let stop = 'STOP';
  for (const frame of parseSseFrames(body)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data) || !Array.isArray(data.candidates)) continue;
    for (const candidate of data.candidates) {
      if (!isRecord(candidate)) continue;
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') text += part.text;
      }
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      usage = {
        input: numOrNull(data.usageMetadata.promptTokenCount) ?? usage.input,
        output: numOrNull(data.usageMetadata.candidatesTokenCount) ?? usage.output,
      };
    }
  }
  return { text, usage, stop };
}

// ---------------------------------------------------------------------------
// Client answers, rendered from one provider answer in the client's wire.
// ---------------------------------------------------------------------------

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A provider stop reason as the client's wire wants it. */
function clientStop(providerStop: string, clientWire: 'openai' | 'anthropic'): string {
  const family: Record<string, string> = {
    stop: '0', end_turn: '0', STOP: '0',
    length: '1', max_tokens: '1', MAX_TOKENS: '1',
    tool_calls: '2', function_call: '2', tool_use: '2',
    content_filter: '3', refusal: '3', SAFETY: '3',
  };
  const f = family[providerStop] ?? '0';
  if (clientWire === 'openai') return f === '1' ? 'length' : f === '2' ? 'tool_calls' : f === '3' ? 'content_filter' : 'stop';
  return f === '1' ? 'max_tokens' : f === '2' ? 'tool_use' : f === '3' ? 'refusal' : 'end_turn';
}

function renderOpenAI(answer: ProviderAnswer, model: string, stream: boolean): { status: number; headers: Record<string, string>; body: string } {
  const stop = clientStop(answer.stop, 'openai');
  if (!stream) {
    return {
      status: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        id: 'chatcmpl-sim',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
        usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text, 13)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' },
    body: frames.join(''),
  };
}

function renderAnthropic(answer: ProviderAnswer, model: string, stream: boolean): { status: number; headers: Record<string, string>; body: string } {
  const stop = clientStop(answer.stop, 'anthropic');
  if (!stream) {
    return {
      status: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
      }),
    };
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text, 13)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' },
    body: frames.join(''),
  };
}


// ---------------------------------------------------------------------------
// Routing: the best free model that meets the request, with budget-aware
// spreading and retry-then-failover on a refusal.
// ---------------------------------------------------------------------------

interface ModelCandidate {
  provider: ProviderEntry;
  row: RatingRow;
}

/** Whether a quality-table row meets a request's requirements. */
function meetsRequirements(row: RatingRow, req: Requirement): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && (row.context_length === null || row.context_length < req.min_context)) return false;
  return true;
}

/** Per-provider call counts (every call, success or refusal: the provider counts both). */
const providerCalls = new Map<string, number>();

function callsOf(providerId: string): number {
  return providerCalls.get(providerId) ?? 0;
}

function noteCall(providerId: string): void {
  providerCalls.set(providerId, callsOf(providerId) + 1);
}

/**
 * The eligible models for one request, in attempt order: rating descending, ties broken by the
 * less-used provider, then provider and model id (deterministic). Models whose provider is under
 * its soft cap come first, so the primaries spread over providers; the rest stay in the chain as
 * failovers (their real budget may be above the cap).
 */
function candidatesFor(brief: Brief, req: Requirement, jsonNeedsOpenWire: boolean): ModelCandidate[] {
  const eligible: ModelCandidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!meetsRequirements(row, req)) continue;
    const provider = brief.providers.find((p) => p.id === row.provider);
    if (provider === undefined) continue;
    if (jsonNeedsOpenWire && provider.wire === 'anthropic') continue;
    eligible.push({ provider, row });
  }
  eligible.sort((a, b) => {
    if (b.row.rating_milli !== a.row.rating_milli) return b.row.rating_milli - a.row.rating_milli;
    const ca = callsOf(a.provider.id);
    const cb = callsOf(b.provider.id);
    if (ca !== cb) return ca - cb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  const under = eligible.filter((c) => callsOf(c.provider.id) < (SOFT_CAP[c.provider.id] ?? 0));
  const at = eligible.filter((c) => callsOf(c.provider.id) >= (SOFT_CAP[c.provider.id] ?? 0));
  return [...under, ...at];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Absolute base URLs from GET /instance, when the env answered (else the brief's paths). */
const instanceBase = new Map<string, string>();

function providerBase(provider: ProviderEntry): string {
  const absolute = instanceBase.get(provider.id);
  if (absolute !== undefined && absolute !== '') return absolute.replace(/\/+$/, '');
  return joinBase(ENV_URL, provider.base);
}

interface DialOk {
  ok: true;
  answer: ProviderAnswer;
}

interface DialFail {
  ok: false;
  status: number;
  code: string;
  terminal: boolean;
}

type DialResult = DialOk | DialFail;

function errorCodeOf(wire: Wire, status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string') return value.error.code;
      if (typeof value.error.type === 'string') return value.error.type;
      if (typeof value.error.status === 'string') return value.error.status;
    }
  } catch {
    // Not JSON: fall through.
  }
  return `http_${status}`;
}

/** One call to one provider's model, in that provider's wire. A 2xx is terminal: its answer is
 *  used (or the request fails if it cannot be parsed); a refusal is returned for failover. */
async function dial(provider: ProviderEntry, model: string, canon: CanonRequest): Promise<DialResult> {
  const base = providerBase(provider);
  const stream = canon.stream;
  let url: string;
  let body: Json;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = canonToOpenAI(canon, model);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    body = canonToAnthropic(canon, model);
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    body = canonToGemini(canon);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 502, code: 'upstream_unreachable', terminal: false };
  }
  noteCall(provider.id);
  const text = await res.text();
  if (!res.ok) {
    const status = res.status;
    return { ok: false, status, code: errorCodeOf(provider.wire, status, text), terminal: false };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (answer === null) return { ok: false, status: 502, code: 'bad_upstream_answer', terminal: true };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Serving one client request.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
let manifestIndex = 0;

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

function requirementsFor(body: Json): { req: Requirement; jsonNeedsOpenWire: boolean } {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) {
    const jsonNeedsOpenWire =
      isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
    return { req: entry.requires, jsonNeedsOpenWire };
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const json = format === 'json_object' || format === 'json_schema';
  return {
    req: {
      tools: Array.isArray(body.tools) && body.tools.length > 0,
      vision,
      json,
    },
    jsonNeedsOpenWire: json,
  };
}

function failureHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  const text = JSON.stringify(rows);
  if (text.length > 8192) return {};
  return { 'x-makefun-upstream-failures': text };
}

function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** Serve one client request body on its wire. Returns the response to send. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json): Promise<{ status: number; headers: Record<string, string>; body: string }> {
  const canon = clientWire === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const { req, jsonNeedsOpenWire } = requirementsFor(body);
  const candidates = candidatesFor(brief, req, jsonNeedsOpenWire);
  const failures: FailureRow[] = [];
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
    // One retry of the same model: a refusal on a provider's failure schedule is a one-off call
    // number, so the next call to the same model usually succeeds; a budget refusal fails twice
    // and the chain moves on.
    for (let retry = 0; retry < 2 && attempts < MAX_ATTEMPTS_PER_REQUEST; retry++) {
      attempts += 1;
      const result = await dial(candidate.provider, candidate.row.model, canon);
      if (result.ok) {
        const rendered =
          clientWire === 'openai'
            ? renderOpenAI(result.answer, candidate.row.model, canon.stream)
            : renderAnthropic(result.answer, candidate.row.model, canon.stream);
        return { status: rendered.status, headers: { ...rendered.headers, ...failureHeader(failures) }, body: rendered.body };
      }
      if (result.terminal) {
        return errorResponse(clientWire, 502, 'the provider answered a body that cannot be read', failures);
      }
      if (isRefusal(result.status)) {
        failures.push({ status: result.status, code: result.code, model: candidate.row.model });
        if (retry === 0) continue;
        break;
      }
      // A 400/401 is this entry's own error: do not retry the model, move on.
      break;
    }
  }
  return errorResponse(clientWire, 502, 'no free model on the directory answered this request', failures);
}

function errorResponse(clientWire: 'openai' | 'anthropic', status: number, message: string, failures: FailureRow[]): { status: number; headers: Record<string, string>; body: string } {
  const headers: Record<string, string> = { 'content-type': 'application/json', ...failureHeader(failures) };
  if (clientWire === 'anthropic') {
    return { status, headers, body: JSON.stringify({ type: 'error', error: { type: 'api_error', message } }) };
  }
  return { status, headers, body: JSON.stringify({ error: { message, type: 'api_error', code: null } }) };
}

// ---------------------------------------------------------------------------
// The HTTP server.
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, headers: Record<string, string>, body: string): void {
  res.writeHead(status, { ...headers, 'content-length': String(Buffer.byteLength(body)) });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, message: string): void {
  const body =
    wire === 'anthropic'
      ? JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message } })
      : JSON.stringify({ error: { message, type: 'invalid_request_error', code: null } });
  sendJson(res, status, { 'content-type': 'application/json' }, body);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const rows: { id: string; provider: string; context_length: number | null; tools: boolean; vision: boolean; json: boolean }[] = [];
  for (const p of brief.providers) {
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free) continue;
      rows.push({ id: row.model, provider: p.id, context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json });
    }
  }
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(
      res,
      200,
      { 'content-type': 'application/json' },
      JSON.stringify({
        data: rows.map((m) => ({
          type: 'model',
          id: m.id,
          display_name: m.id,
          created_at: FIXED_ANTHROPIC_CREATED_AT,
          free: true,
          context_length: m.context_length,
          tools: m.tools,
          vision: m.vision,
          json: m.json,
        })),
        has_more: false,
      }),
    );
  } else {
    sendJson(
      res,
      200,
      { 'content-type': 'application/json' },
      JSON.stringify({
        object: 'list',
        data: rows.map((m) => ({
          id: m.id,
          object: 'model',
          created: FIXED_CREATED,
          owned_by: m.provider,
          free: true,
          context_length: m.context_length,
          tools: m.tools,
          vision: m.vision,
          json: m.json,
        })),
      }),
    );
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const wire: 'openai' | 'anthropic' = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) {
      sendError(res, wire, 400, 'the request body is not one JSON object');
      return;
    }
    const answer = await serve(wire, body);
    sendJson(res, answer.status, answer.headers, answer.body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

/** GET /instance for the provider directory with absolute base URLs for dialling. */
async function fetchInstance(): Promise<void> {
  if (ENV_URL === '') return;
  try {
    const res = await fetch(`${ENV_URL}/instance`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) {
      log(`GET /instance answered HTTP ${res.status}; dialling from the brief's paths`);
      return;
    }
    const value: unknown = await res.json();
    if (!isRecord(value) || !Array.isArray(value.providers)) return;
    for (const raw of value.providers) {
      if (!isRecord(raw) || typeof raw.id !== 'string' || typeof raw.base_url !== 'string') continue;
      instanceBase.set(raw.id, raw.base_url);
    }
  } catch (err) {
    log(`GET /instance failed (${(err as Error).message}); dialling from the brief's paths`);
  }
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  await fetchInstance();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    });
  });
  const port = await listen(server);
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels(brief).length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
