"""Deployment settings. The publishable key is safe to ship inside the app:
row-level security in the database decides what a logged-in user may see.
Never put the secret / service-role key here."""

import os

SUPABASE_URL = os.environ.get("CHESS_VISION_SUPABASE_URL", "https://rovlrbjifawlsorrxtso.supabase.co")
SUPABASE_KEY = os.environ.get(
    "CHESS_VISION_SUPABASE_KEY",
    "sb_publishable_mb1dCqK6AXx0xRZak9TvXw_8QcZkhOc",
)
KEYCHAIN_SERVICE = "chess-vision"
