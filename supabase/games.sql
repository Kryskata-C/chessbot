-- Chess Vision: per-user game history uploaded by the desktop app and shown
-- on the website dashboard. Run once in the Supabase SQL editor; safe to re-run.
create table if not exists public.games (
  id                uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  played_at         timestamptz not null default now(),   -- when the game started
  duration_s        integer,
  color             text check (color in ('w', 'b')),
  result            text not null default '*',            -- '1-0' | '0-1' | '1/2-1/2' | '*' (unfinished / not seen)
  termination       text,                                 -- checkmate | resignation | timeout | agreement | stalemate | repetition | insufficient | fifty_moves | abandoned | aborted
  score             real,                                 -- 1 win, 0.5 draw, 0 loss, null unfinished
  plies             integer,
  target_elo        integer,
  opponent_rating   integer,                              -- read off chess.com next to the name
  opponent_estimate integer,                              -- from their moves
  accuracy          real,                                 -- bot best-move %
  acpl              real,
  contested_acpl    real,
  realized_elo      integer,
  resyncs           integer not null default 0,
  platform          text,                                 -- 'mac' | 'win'
  app_version       text,
  pgn               text,
  created_at        timestamptz not null default now()
);
-- Added after the first version of this table; harmless on a fresh one.
alter table public.games add column if not exists termination text;

create index if not exists games_user_played_idx on public.games (user_id, played_at desc);

alter table public.games enable row level security;

drop policy if exists "games: read own or admin" on public.games;
create policy "games: read own or admin" on public.games
  for select using (auth.uid() = user_id or public.is_admin());

drop policy if exists "games: insert own" on public.games;
create policy "games: insert own" on public.games
  for insert with check (auth.uid() = user_id);

drop policy if exists "games: admin delete" on public.games;
create policy "games: admin delete" on public.games
  for delete using (public.is_admin());

-- Sanity check
select count(*) as games from public.games;
