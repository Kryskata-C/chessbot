// Chess Vision — Stripe → Supabase licence sync.
//
// Stripe calls this URL for billing events; we flip profiles.active and set
// profiles.expires_at, which is all the site (sb.js) and the app (auth.py)
// ever look at. Nothing here is reachable from the browser: the function is
// deployed with --no-verify-jwt and only accepts requests carrying a valid
// Stripe signature.
//
// Secrets (supabase secrets set ...):
//   STRIPE_SECRET_KEY       sk_live_... / sk_test_...
//   STRIPE_WEBHOOK_SECRET   whsec_... from the endpoint in the Stripe dashboard
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   injected by Supabase automatically
//
// Events to subscribe the endpoint to:
//   checkout.session.completed   first payment  → active, link customer to profile
//   invoice.paid                 renewals       → active, push expiry forward
//   invoice.payment_failed       renewal failed → inactive
//   customer.subscription.updated  status changes (past_due, unpaid, cancel) → mirror
//   customer.subscription.deleted  cancelled / ended → inactive

import Stripe from "npm:stripe@18";
import { createClient } from "npm:@supabase/supabase-js@2";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "");
const WEBHOOK_SECRET = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";
const db = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false } },
);

// A renewal can land a few hours after the period ends while Stripe retries a
// card; one day of grace stops the app locking someone out over a bank delay.
const GRACE_SECONDS = 24 * 3600;
const ACTIVE_STATUSES = new Set(["active", "trialing"]);

type Patch = { active?: boolean; expires_at?: string | null; stripe_customer_id?: string; stripe_subscription_id?: string | null };

const log = (...a: unknown[]) => console.log("[stripe-webhook]", ...a);
const respond = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const customerId = (c: string | Stripe.Customer | Stripe.DeletedCustomer | null | undefined) =>
  typeof c === "string" ? c : c?.id ?? null;

// Period end lives on the subscription item since API 2025-03-31; older
// versions had it on the subscription itself. Read whichever exists.
function periodEnd(sub: Stripe.Subscription): string | null {
  const item = sub.items?.data?.[0] as (Stripe.SubscriptionItem & { current_period_end?: number }) | undefined;
  const end = item?.current_period_end ?? (sub as unknown as { current_period_end?: number }).current_period_end;
  return end ? new Date((end + GRACE_SECONDS) * 1000).toISOString() : null;
}

async function subscriptionFor(invoice: Stripe.Invoice): Promise<Stripe.Subscription | null> {
  const inv = invoice as Stripe.Invoice & {
    subscription?: string | Stripe.Subscription | null;
    parent?: { subscription_details?: { subscription?: string | Stripe.Subscription | null } | null } | null;
  };
  const ref = inv.parent?.subscription_details?.subscription ?? inv.subscription ?? null;
  if (!ref) return null;
  return typeof ref === "string" ? await stripe.subscriptions.retrieve(ref) : ref;
}

// Find the profile a Stripe object belongs to. Preference order:
//   1. the user id we passed as client_reference_id at checkout
//   2. a profile already linked to this Stripe customer
//   3. the customer's e-mail (matches profiles.email, which mirrors auth.users)
async function findProfile(opts: { userId?: string | null; customer?: string | null; email?: string | null }) {
  const sel = "id, email, role, active, expires_at, stripe_customer_id";
  if (opts.userId) {
    const { data } = await db.from("profiles").select(sel).eq("id", opts.userId).maybeSingle();
    if (data) return data;
  }
  if (opts.customer) {
    const { data } = await db.from("profiles").select(sel).eq("stripe_customer_id", opts.customer).maybeSingle();
    if (data) return data;
  }
  if (opts.email) {
    const { data } = await db.from("profiles").select(sel).ilike("email", opts.email).maybeSingle();
    if (data) return data;
  }
  return null;
}

async function apply(profileId: string, patch: Patch, why: string) {
  const { error } = await db.from("profiles").update(patch).eq("id", profileId);
  if (error) throw new Error(`profiles update failed (${why}): ${error.message}`);
  log(why, profileId, patch);
}

async function customerEmail(customer: string | null): Promise<string | null> {
  if (!customer) return null;
  const c = await stripe.customers.retrieve(customer);
  return c.deleted ? null : c.email ?? null;
}

async function handle(event: Stripe.Event) {
  switch (event.type) {
    case "checkout.session.completed": {
      const s = event.data.object as Stripe.Checkout.Session;
      if (s.mode !== "subscription" || s.payment_status !== "paid") return "ignored: not a paid subscription checkout";
      const customer = customerId(s.customer);
      const subId = typeof s.subscription === "string" ? s.subscription : s.subscription?.id ?? null;
      const profile = await findProfile({ userId: s.client_reference_id, customer, email: s.customer_details?.email ?? s.customer_email });
      if (!profile) throw new Error(`no profile for checkout ${s.id} (ref=${s.client_reference_id}, email=${s.customer_details?.email})`);
      const sub = subId ? await stripe.subscriptions.retrieve(subId) : null;
      await apply(profile.id, {
        active: true,
        expires_at: sub ? periodEnd(sub) : null,
        stripe_customer_id: customer ?? undefined,
        stripe_subscription_id: subId,
      }, "checkout completed");
      return "activated";
    }

    case "invoice.paid": {
      const inv = event.data.object as Stripe.Invoice;
      const customer = customerId(inv.customer);
      const sub = await subscriptionFor(inv);
      if (!sub) return "ignored: invoice without subscription";
      const profile = await findProfile({ customer, email: inv.customer_email ?? await customerEmail(customer) });
      if (!profile) throw new Error(`no profile for invoice ${inv.id} (customer=${customer})`);
      await apply(profile.id, {
        active: true,
        expires_at: periodEnd(sub),
        stripe_customer_id: customer ?? undefined,
        stripe_subscription_id: sub.id,
      }, "invoice paid");
      return "renewed";
    }

    case "invoice.payment_failed": {
      const inv = event.data.object as Stripe.Invoice;
      const customer = customerId(inv.customer);
      const profile = await findProfile({ customer, email: inv.customer_email });
      if (!profile) return `ignored: no profile for customer ${customer}`;
      if (profile.role !== "user") return "ignored: friend/admin accounts do not depend on billing";
      await apply(profile.id, { active: false }, "payment failed");
      return "deactivated";
    }

    case "customer.subscription.updated": {
      const sub = event.data.object as Stripe.Subscription;
      const customer = customerId(sub.customer);
      const profile = await findProfile({ customer, email: await customerEmail(customer) });
      if (!profile) return `ignored: no profile for customer ${customer}`;
      const on = ACTIVE_STATUSES.has(sub.status);
      await apply(profile.id, {
        active: on,
        expires_at: on ? periodEnd(sub) : profile.expires_at,
        stripe_customer_id: customer ?? undefined,
        stripe_subscription_id: sub.id,
      }, `subscription ${sub.status}`);
      return on ? "active" : "inactive";
    }

    case "customer.subscription.deleted": {
      const sub = event.data.object as Stripe.Subscription;
      const customer = customerId(sub.customer);
      const profile = await findProfile({ customer });
      if (!profile) return `ignored: no profile for customer ${customer}`;
      await apply(profile.id, { active: false, stripe_subscription_id: null }, "subscription deleted");
      return "deactivated";
    }

    default:
      return `ignored: ${event.type}`;
  }
}

Deno.serve(async (req) => {
  if (req.method !== "POST") return respond(405, { error: "POST only" });
  const sig = req.headers.get("stripe-signature");
  if (!sig || !WEBHOOK_SECRET) return respond(400, { error: "missing signature or webhook secret" });

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(await req.text(), sig, WEBHOOK_SECRET);
  } catch (e) {
    log("bad signature", (e as Error).message);
    return respond(400, { error: "invalid signature" });
  }

  // Idempotency: Stripe re-sends on any non-2xx and sometimes on success too.
  const { error: dupErr } = await db.from("stripe_events").insert({ id: event.id, type: event.type });
  if (dupErr) {
    if (dupErr.code === "23505") return respond(200, { ok: true, result: "duplicate, already applied" });
    log("stripe_events insert failed", dupErr.message);   // table missing? still process the event
  }

  try {
    const result = await handle(event);
    return respond(200, { ok: true, result });
  } catch (e) {
    log("error", event.type, (e as Error).message);
    await db.from("stripe_events").delete().eq("id", event.id);   // let Stripe retry it
    return respond(500, { ok: false, error: (e as Error).message });
  }
});
