create extension if not exists vector;

drop table if exists public.session_documents cascade;
drop table if exists public.messages cascade;
drop table if exists public.documents cascade;
drop table if exists public.sessions cascade;

create table public.sessions (
  id uuid primary key,
  title text not null default '새 세션',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.messages (
  id bigserial primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  seq integer not null default 0,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create table public.documents (
  id bigserial primary key,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  content_hash text not null unique,
  embedding vector(1536) not null,
  created_at timestamptz not null default now()
);

create table public.session_documents (
  session_id uuid not null references public.sessions(id) on delete cascade,
  document_id bigint not null references public.documents(id) on delete cascade,
  source_name text,
  created_at timestamptz not null default now(),
  primary key (session_id, document_id)
);

create index idx_messages_session_seq on public.messages(session_id, seq);
create index idx_sessions_updated_at on public.sessions(updated_at desc);
create index idx_session_docs_session on public.session_documents(session_id);
create index idx_documents_embedding on public.documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_sessions_updated_at
before update on public.sessions
for each row execute function public.set_updated_at();

alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.documents enable row level security;
alter table public.session_documents enable row level security;

create policy "open sessions all" on public.sessions for all to anon, authenticated using (true) with check (true);
create policy "open messages all" on public.messages for all to anon, authenticated using (true) with check (true);
create policy "open documents all" on public.documents for all to anon, authenticated using (true) with check (true);
create policy "open session_documents all" on public.session_documents for all to anon, authenticated using (true) with check (true);

create or replace function public.match_session_documents(
  target_session_id uuid,
  query_embedding vector(1536),
  match_threshold float default 0.55,
  match_count int default 8
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  join public.session_documents sd on sd.document_id = d.id
  where sd.session_id = target_session_id
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_session_documents(uuid, vector, float, int) to anon, authenticated;
