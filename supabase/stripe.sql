-- Chess Vision: Stripe billing columns + webhook idempotency table.
-- Run once in the Supabase SQL editor; safe to re-run.
-- The stripe-webhook Edge Function (supabase/functions/stripe-webhook) writes
-- these with the service-role key; the site and app only ever read
-- profiles.active / profiles.expires_at, which are unchanged.

alter table public.profiles add column if not exists stripe_customer_id     text;
alter table public.profiles add column if not exists stripe_subscription_id text;

create unique index if not exists profiles_stripe_customer_idx
  on public.profiles (stripe_customer_id) where stripe_customer_id is not null;

-- Stripe retries webhooks; remember every event id we have already applied.
create table if not exists public.stripe_events (
  id          text primary key,             -- evt_...
  type        text not null,
  received_at timestamptz not null default now()
);
alter table public.stripe_events enable row level security;   -- no policies: service role only

-- Sanity check
select count(*) filter (where stripe_customer_id is not null) as linked_to_stripe,
       count(*) filter (where active) as active
  from public.profiles;
