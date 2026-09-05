-- Chess Vision: add the `friend` role (unlimited access, no admin powers).
-- Run once in the Supabase SQL editor. Safe to re-run.
-- Works whether profiles.role is a text column with a CHECK constraint or an enum type.
do $$
declare
  udt text;
  con record;
begin
  select udt_name into udt from information_schema.columns
   where table_schema = 'public' and table_name = 'profiles' and column_name = 'role';
  if udt is null then
    raise exception 'public.profiles.role not found';
  end if;

  if exists (select 1 from pg_type t join pg_namespace n on n.oid = t.typnamespace
             where t.typname = udt and t.typtype = 'e') then
    -- enum column: add the value if missing
    if not exists (select 1 from pg_enum e join pg_type t on t.oid = e.enumtypid
                   where t.typname = udt and e.enumlabel = 'friend') then
      execute format('alter type %I add value %L', udt, 'friend');
    end if;
  else
    -- text column: replace any CHECK constraint that mentions role
    for con in select conname from pg_constraint
                where conrelid = 'public.profiles'::regclass and contype = 'c'
                  and pg_get_constraintdef(oid) ilike '%role%'
    loop
      execute format('alter table public.profiles drop constraint %I', con.conname);
    end loop;
    alter table public.profiles
      add constraint profiles_role_check check (role in ('user', 'friend', 'admin'));
  end if;
end $$;

-- is_admin() is unchanged: friends are never admins.
-- Sanity check:
select role, count(*) from public.profiles group by role;
