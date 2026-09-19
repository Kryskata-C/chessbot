# Stripe → Chess Vision licence sync

One Edge Function keeps `profiles.active` / `profiles.expires_at` in step with
Stripe. The site and the app never talk to Stripe themselves.

```
pricing button ──► Stripe Payment Link ──► redirect thanks.html
                          │
                          └─ webhook ──► stripe-webhook (this function) ──► profiles
```

## One-time setup

### 1. Database
Run `supabase/stripe.sql` in the SQL editor (adds `stripe_customer_id`,
`stripe_subscription_id`, and the `stripe_events` idempotency table).

### 2. Stripe dashboard
1. **Product** → "Chess Vision", description *"Chess training and analysis
   overlay. Monthly subscription."* Recurring price €10.99 / month.
2. **Payment Link** for that price:
   - Collect customer e-mail (default).
   - After payment → *Don't show confirmation page*, redirect to
     `https://cool-dango-7a52fd.netlify.app/thanks.html`
     (use the custom domain once there is one).
   - Copy the link (`https://buy.stripe.com/...`) → `CV.STRIPE.PAY_LINK` in
     `~/chess-vision-site/sb.js`.
3. **Customer portal**: Settings → Billing → Customer portal → activate,
   allow cancelling and updating the payment method. Copy the login link
   (`https://billing.stripe.com/p/login/...`) → `CV.STRIPE.PORTAL`.
4. **Webhook**: Developers → Webhooks → Add endpoint
   - URL `https://rovlrbjifawlsorrxtso.supabase.co/functions/v1/stripe-webhook`
   - Events: `checkout.session.completed`, `invoice.paid`,
     `invoice.payment_failed`, `customer.subscription.updated`,
     `customer.subscription.deleted`
   - Copy the signing secret (`whsec_...`).

### 3. Deploy the function
```sh
brew install supabase/tap/supabase          # already installed
cd ~/chessbot
supabase login
supabase link --project-ref rovlrbjifawlsorrxtso
supabase secrets set STRIPE_SECRET_KEY=sk_live_... STRIPE_WEBHOOK_SECRET=whsec_...
supabase functions deploy stripe-webhook --no-verify-jwt
```
`--no-verify-jwt` is required: Stripe does not send a Supabase JWT. The
Stripe signature check is the authentication.

### 4. Site
Fill `CV.STRIPE.PAY_LINK`, `CV.STRIPE.PORTAL` and `CV.SUPPORT_EMAIL` in
`sb.js`, push to main, Netlify deploys.

## Testing (test mode first)
Use the test-mode keys, a test-mode Payment Link, and card `4242 4242 4242 4242`.
Then in Stripe → Webhooks → the endpoint → check each delivery is `200`;
the response body says what happened (`activated`, `renewed`, `duplicate…`).
Logs: Supabase dashboard → Edge Functions → stripe-webhook → Logs.

## What each event does
| Event | Effect on the profile |
|---|---|
| `checkout.session.completed` | `active = true`, `expires_at` = period end + 1 day grace, links customer/subscription ids |
| `invoice.paid` | same, pushes `expires_at` forward each month |
| `invoice.payment_failed` | `active = false` (only for `role = user`) |
| `customer.subscription.updated` | mirrors the status: active/trialing → on, anything else → off |
| `customer.subscription.deleted` | `active = false`, clears the subscription id |

The profile is matched by (in order) the `client_reference_id` the site
attaches at checkout (= Supabase user id), an already-linked
`stripe_customer_id`, then the customer's e-mail. Friend and admin accounts
never lose access through billing events (see `CV.licence` / `auth.py`).
