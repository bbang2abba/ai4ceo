-- multi-users-ref.py 전용 Supabase 스키마
-- Supabase Dashboard → SQL Editor 에서 한 번에 실행하세요.
-- UTF-8
--
-- 필요: Authentication → Providers → Email, Confirm email ON
-- 앱에서는 SUPABASE_ANON_KEY + 로그인 사용자 JWT 로 호출해야 RLS 가 적용됩니다.

-- ---------------------------------------------------------------------------
-- (선택) 완전 초기화 후 재설치할 때만 주석 해제
-- ---------------------------------------------------------------------------
-- DROP TRIGGER IF EXISTS delete_embeddings_on_session_delete ON public.sessions;
-- DROP TRIGGER IF EXISTS update_sessions_updated_at ON public.sessions;
-- DROP FUNCTION IF EXISTS public.delete_session_embeddings();
-- DROP FUNCTION IF EXISTS public.update_updated_at_column();
-- DROP FUNCTION IF EXISTS public.match_documents(vector, double precision, integer, uuid, text);
-- DROP TABLE IF EXISTS public.embeddings CASCADE;
-- DROP TABLE IF EXISTS public.sessions CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- embeddings: PDF 청크 + 벡터 (사용자·세션별)
-- Python: upsert on_conflict = user_id,session_id,file_name,chunk_index
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    session_id text NOT NULL,
    file_name text NOT NULL,
    chunk_index integer NOT NULL,
    chunk_text text NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT embeddings_user_session_file_chunk_uniq UNIQUE (user_id, session_id, file_name, chunk_index)
);

CREATE INDEX IF NOT EXISTS embeddings_user_id_idx ON public.embeddings (user_id);
CREATE INDEX IF NOT EXISTS embeddings_session_id_idx ON public.embeddings (session_id);
CREATE INDEX IF NOT EXISTS embeddings_user_session_idx ON public.embeddings (user_id, session_id);

-- 데이터가 쌓인 뒤 성능이 필요하면 ivfflat / hnsw 인덱스 추가 권장 (빈 테이블이면 생략 가능)
CREATE INDEX IF NOT EXISTS embeddings_vector_ivfflat_idx
    ON public.embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- sessions: 채팅 세션 메타 + JSON 문자열 컬럼
-- Python: id = session UUID 문자열, session_id 동일 값, upsert on_conflict = id
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sessions (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    session_id text NOT NULL,
    title text,
    chat_history text,
    conversation_memory text,
    processed_files text,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT sessions_user_session_id_uniq UNIQUE (user_id, session_id)
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON public.sessions (user_id);
CREATE INDEX IF NOT EXISTS sessions_updated_at_idx ON public.sessions (updated_at DESC);

-- ---------------------------------------------------------------------------
-- 세션 삭제 시 해당 세션 임베딩 자동 삭제
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.delete_session_embeddings()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM public.embeddings
    WHERE user_id = OLD.user_id
      AND session_id = OLD.session_id;
    RETURN OLD;
END;
$$;

DROP TRIGGER IF EXISTS delete_embeddings_on_session_delete ON public.sessions;
CREATE TRIGGER delete_embeddings_on_session_delete
    AFTER DELETE ON public.sessions
    FOR EACH ROW
    EXECUTE FUNCTION public.delete_session_embeddings();

-- ---------------------------------------------------------------------------
-- updated_at 자동 갱신
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.touch_sessions_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_sessions_updated_at ON public.sessions;
CREATE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON public.sessions
    FOR EACH ROW
    EXECUTE FUNCTION public.touch_sessions_updated_at();

-- ---------------------------------------------------------------------------
-- RPC: match_documents — Python SessionRetriever 가 호출
-- 파라미터 이름: query_embedding, match_threshold, match_count, user_id_filter, session_id_filter
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.match_documents(
    query_embedding vector(1536),
    match_threshold double precision DEFAULT 0.7,
    match_count integer DEFAULT 10,
    user_id_filter uuid DEFAULT NULL,
    session_id_filter text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    user_id uuid,
    session_id text,
    file_name text,
    chunk_index integer,
    chunk_text text,
    metadata jsonb,
    similarity double precision
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.user_id,
        e.session_id,
        e.file_name,
        e.chunk_index,
        e.chunk_text,
        e.metadata,
        (1 - (e.embedding <=> query_embedding))::double precision AS similarity
    FROM public.embeddings e
    WHERE
        (user_id_filter IS NULL OR e.user_id = user_id_filter)
        AND (session_id_filter IS NULL OR e.session_id = session_id_filter)
        AND (1 - (e.embedding <=> query_embedding)) >= match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- PostgREST RPC 호출 허용
GRANT EXECUTE ON FUNCTION public.match_documents(vector, double precision, integer, uuid, text) TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_documents(vector, double precision, integer, uuid, text) TO service_role;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE public.embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "embeddings_select_own" ON public.embeddings;
DROP POLICY IF EXISTS "embeddings_insert_own" ON public.embeddings;
DROP POLICY IF EXISTS "embeddings_update_own" ON public.embeddings;
DROP POLICY IF EXISTS "embeddings_delete_own" ON public.embeddings;

DROP POLICY IF EXISTS "sessions_select_own" ON public.sessions;
DROP POLICY IF EXISTS "sessions_insert_own" ON public.sessions;
DROP POLICY IF EXISTS "sessions_update_own" ON public.sessions;
DROP POLICY IF EXISTS "sessions_delete_own" ON public.sessions;

CREATE POLICY "embeddings_select_own" ON public.embeddings FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "embeddings_insert_own" ON public.embeddings FOR INSERT TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "embeddings_update_own" ON public.embeddings FOR UPDATE TO authenticated
    USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "embeddings_delete_own" ON public.embeddings FOR DELETE TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "sessions_select_own" ON public.sessions FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "sessions_insert_own" ON public.sessions FOR INSERT TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "sessions_update_own" ON public.sessions FOR UPDATE TO authenticated
    USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "sessions_delete_own" ON public.sessions FOR DELETE TO authenticated
    USING (auth.uid() = user_id);

-- API(PostgREST) 접근
GRANT SELECT, INSERT, UPDATE, DELETE ON public.embeddings TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sessions TO authenticated;
